import argparse
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
import sklearn.metrics
from sklearn.metrics import roc_auc_score
import time
import os
import re
import pickle
from datetime import datetime
from pathlib import Path

def sample_frames(frames_path, sampled_num):
    all_frames = sorted([
        os.path.join(frames_path, f)
        for f in os.listdir(frames_path)
        if f.lower().endswith('.jpg')
    ])

    total_frames = len(all_frames)
    if sampled_num >= total_frames:
        return all_frames

    indices = [int(i * total_frames / sampled_num) for i in range(sampled_num)]

    indices = [min(idx, total_frames - 1) for idx in indices]

    sampled_frames = [all_frames[idx] for idx in indices]

    return sampled_frames

def extract_accident_label(llm_output: str):
    """
    Extract detection label from LLM output.
    Handles both single and double quotes.
    If not found, defaults to: {'Abnormal Behavior': 'No'}
    """
    pattern = r"""\[\s*['"]Abnormal Behavior['"]\s*:\s*['"](Yes|No)['"]\s*\]"""
    match = re.search(pattern, llm_output, re.IGNORECASE)

    if match:
        accident = match.group(1).capitalize()
        return {'Abnormal Behavior': accident}
    else:
        return {'Abnormal Behavior': 'No'}

def inference(processor, model, messages, max_new_tokens, temperature, top_p):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device="cuda")
    input_len = inputs["input_ids"].shape[-1]

    # Inference
    with torch.inference_mode():
        generation = model.generate(**inputs,
                                    max_new_tokens=max_new_tokens,
                                    temperature = temperature,
                                    top_p = top_p)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded

def message_template(img_list):
    sys_input = """You are a helpful and intelligent assistant. 
    You will be provided with a sequence of consecutive frames extracted from a traffic video captured by a Dash Cam (Front windshield camera). 
    Your task is to analyze these frames based on the user's request.
    Specifically, determine whether any abnormal traffic behavior occurs within the frame sequence (e.g., collisions, traffic accidents, or driving against traffic). 
    You must include the prediction result at the end in the following format: ['Abnormal Behavior': 'Yes' or 'No']"""
    user_input_3 = """Describe the given frames from three perspectives—road network, traffic participants, and their behaviors—and determine whether any abnormal traffic behavior occurs (e.g., accidents, driving in the wrong direction, or other irregular actions).
    Be sure to provide the prediction result at the end using the following format: ['Abnormal Behavior': 'Yes' or 'No']"""
    content_list = [{"type": "image", "image": img} for img in img_list]
    content_list.append({"type": "text", "text": user_input_3})
    messages = [
        {"role": "system",
         "content": [{"type": "text", "text": sys_input}]
         },
        {
            "role": "user",
            "content": content_list
        }
    ]
    return messages


def load_frame_sequences(frames_path, n_frames=6, stride=5):
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else -1

    frame_files = sorted(
        [f for f in os.listdir(frames_path) if f.endswith('.jpg')],
        key=extract_number
    )

    all_sequences = []
    i = 0
    while i + n_frames <= len(frame_files):
        window = frame_files[i:i + n_frames]
        sequence = [os.path.join(frames_path, f) for f in window]
        all_sequences.append(sequence)
        i += stride

    if all_sequences and len(all_sequences[-1]) < 3:
        all_sequences = all_sequences[:-1]

    return all_sequences

def evaluation(model, processor, args):
    start_time = time.time()
    detected_num = 0
    failed_videos = []

    validation_path = Path(args.test)
    folder_paths = [f for f in validation_path.iterdir() if f.is_dir()]
    folder_paths_str = [str(f) for f in folder_paths]

    case_num = len(folder_paths_str)

    # go through all videos
    # prepare data
    for video_path in folder_paths_str:
        flag = 0
        print('=========================================')
        print('Work on video id: ', video_path.split('/')[-1])

        print('First, let LLM inference on the sampled frames:')
        # sample several frames
        sampled_frames = sample_frames(video_path, args.sampled_num)
        # first time detection (on sampled frames)
        messages = message_template(sampled_frames)
        # inference
        model_output = inference(processor, model, messages, args.max_new_tokens, args.temperature, args.top_p)
        llm_fmt_res = extract_accident_label(model_output)
        print('-----------------------------------------')
        print('LLM outputs using frames: ')
        print(model_output)
        print('Extracted results: ')
        print(llm_fmt_res)
        print('-----------------------------------------')
        if llm_fmt_res['Abnormal Behavior'] == 'Yes':
            flag = 1
            # successfully detected accidents
            detected_num += 1
        else:
            # loop all frames
            # prepare message
            image_list = load_frame_sequences(frames_path=video_path, n_frames=args.n_frames, stride=args.stride)
            print('Extracted clips No.: ', len(image_list))
            for clip_frames in image_list:
                messages = message_template(clip_frames)
                # inference
                model_output = inference(processor, model, messages, args.max_new_tokens, args.temperature, args.top_p)
                llm_fmt_res = extract_accident_label(model_output)
                print('-----------------------------------------')
                print('LLM outputs using frames: ')
                print(model_output)
                print('Extracted results: ')
                print(llm_fmt_res)
                print('-----------------------------------------')
                if llm_fmt_res['Abnormal Behavior'] == 'Yes':
                    flag = 1
                    break  # stop for
                else:
                    flag = 0
            if flag == 1:
                # successfully detected accidents
                detected_num += 1
            else:
                # store failed videos
                failed_videos.append(video_path)
    end_time = time.time()
    print("Gemma3-4b Zero-shot sampled videos")
    print('Average inference time for a case: ', (end_time - start_time) / case_num, ' s')
    print(f"Evaluation metric - DR: {(detected_num/case_num*100)} %")
    print(f"Evaluation metric - FAR: {((case_num-detected_num) / case_num * 100)} %")
    print('-----------------------------------------')
    print("Wrongly predicted videos: ")
    for i in failed_videos:
        print(i)


def main():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--model_path',
                        default='./gemma-3-4b-it',
                        type=str,
                        help='Local path of your pretrained model')
    parser.add_argument('--max_new_tokens',
                        default=512,
                        help='The maximum number of new tokens generated (i.e. the maximum length of the model output)')
    parser.add_argument('--temperature',
                        default=0.5,
                        help='The smaller the parameter value, the more certain the model will return.')
    parser.add_argument('--top_p',
                        default=0.95,
                        help='Nucleus sampling, the smaller the parameter value, the more certain the model will return.')
    parser.add_argument('--n_frames',
                        default=7,
                        type=int,
                        help='No of frames in a query.')
    parser.add_argument('--stride',
                        default=6,
                        type=int,
                        help='Sliding window')
    parser.add_argument('--sampled_num',
                        default=7,
                        type=int,
                        help='No. frames sampled from videos')

    # Dataset parameters
    parser.add_argument('--test',
                        default='/home/kris/benchmarks/DoTA/validation',
                        type=str)
    args = parser.parse_args()

    # Load model
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)

    # Evaluate
    # Only report performance on the testing data
    print('--------Evaluate on testing set now!----------')
    evaluation(model, processor, args)


if __name__ == '__main__':
    main()