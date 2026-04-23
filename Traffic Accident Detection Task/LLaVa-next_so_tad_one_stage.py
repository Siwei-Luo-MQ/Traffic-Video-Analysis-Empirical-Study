import argparse
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# import flash_attn
from transformers import BitsAndBytesConfig
import sklearn.metrics
from sklearn.metrics import roc_auc_score
import time
import os
import os.path
import re
import pickle
import json
from datetime import datetime


def message_template(img_list):
    sys_input = """You are a helpful and intelligent assistant. You will be provided with a sequence of consecutive frames extracted from a traffic surveillance video captured by a roadside camera. Frames are indexed starting from 0.
    Your task is to analyze these frames based on the user's request.
    Specifically, determine whether a traffic accident occurs within the frame sequence. If an accident is detected, also identify and report the index of the frame where the accident begins.
    You must include accident prediction results at the end in the following format: ['Accident': 'Yes' or 'No', 'Frame': 'N/A' or index]"""
    user_input_1 = """Here's an example for you~
    Q: 
    These images are consecutive frames extracted from a traffic surveillance video, forming a short video clip. Please describe this clip from the following three aspects:
    1. The road network (e.g., intersections, straight roads, curves, etc.),
    2. The traffic participants (e.g., sedans, buses, bicycles, pedestrians, etc.), and
    3. The motion behaviors of the traffic participants.
    Lastly, based on the movement trajectories of the traffic participants, determine whether a traffic accident occurred in the clip.
    Let's think step by step.
    ----------
    A:
    There are 5 frames. Firstly, let's check out the road network~
    ## Road Network
    1. Intersection : The scene shows a controlled intersection with traffic lights. There are marked lanes and pedestrian crosswalks, indicating an organized road network.

    Next, I need to identify the traffic participants and their behaviors~
    ## Traffic Participants & Behaviors
    1. A blue pickup is seen crossing the intersection from the right side of the frame.
    2. A red truck is moving slowly near the intersection on the main road.
    3. A white truck is parked or moving slowly on the right side of the frame, near the sidewalk.
    4. A motorcycle is visible on the right side of the frame, near the white truck.

    Lastly, based on the observation, i need to find out whether a traffic accident occurred in the clip.
    ## Determination of Traffic Accident
    The blue pickup truck collided with the red truck in the intersection. The index of accident initial frame is 2. (starting from 0)
    ## Final Output
    ['Accident': 'Yes', 'Frame': 2]
    """
    ass_input_1 = """Understood. I need to first describe the given frames from the perspective of road network, traffic participants and motion behaviors of the traffic participants, and then determine whether there is an accident in the video. If there is an accident, then output the starting frame of the accident. 
    In this example, a blue pickup collided with a red truck, and the starting frame of the accident is 2. I need to include the formatted results in the end: ['Accident': 'Yes', 'Frame': 2]"""

    user_input_2 = """Describe these frames from the three perspectives and determine whether there is an accident and the initial frame of the accident. Remember to think step by step."""

    ass_input_2 = """There are five frames in your new example.
    ## Road Network
    Type of Road : The scene depicts a multi-lane urban road at night. The road appears to be a divided highway with multiple lanes in each direction.
    There are at least two lanes visible in each direction.
    ## Traffic Participants & Behaviors
    1. A bus is prominently visible in the right lane, moving away from the camera.
    2. Several cars are visible in the left lane, also moving away from the camera. Their headlights indicate they are traveling in the same direction as the bus.
    3. No other types of vehicles (e.g., motorcycles, bicycles) are clearly visible in these frames.
    4. No pedestrians are visible in the frames. The sidewalks appear empty.
    ## Determination of Traffic Accident
    1. All vehicles are moving smoothly and predictably within their respective lanes.
    2. There are no visible deformations, debris, smoke, or other indicators of an accident.
    ## Final Output:
    ['Accident': 'No', 'Frame': 'N/A']
    """
    user_input_3 = """Describe these new given frames from the three perspectives and determine whether there is an accident and the initial frame of the accident. Please think step by step."""

    content_list = [{"type": "image", "image": img} for img in img_list]
    content_list.append({"type": "text", "text": user_input_3})
    messages = [
        {"role": "system",
         "content": [{"type": "text", "text": sys_input}]
         },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "./Benchmark/SO_TAD/Extracted_frames/79/0064.jpg"},
                {"type": "image", "image": "./Benchmark/SO_TAD/Extracted_frames/79/0065.jpg"},
                {"type": "image", "image": "./Benchmark/SO_TAD/Extracted_frames/79/0066.jpg"},
                {"type": "image", "image": "./Benchmark/SO_TAD/Extracted_frames/79/0067.jpg"},
                {"type": "image", "image": "./Benchmark/SO_TAD/Extracted_frames/79/0068.jpg"},
                {"type": "text", "text": user_input_1},
            ]
        },
        {"role": "assistant",
         "content": [{"type": "text", "text": ass_input_1}]
         },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "./Benchmark/SO_TAD/Extracted_frames/1449/0023.jpg"},
                {"type": "image", "image": "./Benchmark/SO_TAD/Extracted_frames/1449/0024.jpg"},
                {"type": "image", "image": "./Benchmark/SO_TAD/Extracted_frames/1449/0025.jpg"},
                {"type": "image", "image": "./Benchmark/SO_TAD/Extracted_frames/1449/0026.jpg"},
                {"type": "image", "image": "./Benchmark/SO_TAD/Extracted_frames/1449/0027.jpg"},
                {"type": "text", "text": user_input_2},
            ]
        },
        {"role": "assistant",
         "content": [{"type": "text", "text": ass_input_2}]
         },
        {
            "role": "user",
            "content": content_list
        }
    ]
    return messages

def load_labels(label_file):
    accident_labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                video_name, frame_idx = parts[0], int(parts[1])
                accident_labels[video_name] = frame_idx
    return accident_labels

def extract_labels(video_path, output_dir, labels):
    results = []
    video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
    for video_file in video_files:
        video_file_path = os.path.join(video_path, video_file)
        frame_output_dir = os.path.join(output_dir, os.path.splitext(video_file)[0])
        accident_flag = 1 if video_file in labels else 0
        accident_frame = labels.get(video_file, -1)
        results.append([video_file_path, frame_output_dir, accident_flag, accident_frame])
    return results

# def load_frame_sequences(frames_path, n_frames=6, stride=5):
#     frame_files = sorted(
#         [f for f in os.listdir(frames_path) if f.endswith('.jpg')],
#         key=lambda x: int(os.path.splitext(x)[0])
#     )
#
#     all_sequences = []
#     i = 0
#     while i + n_frames <= len(frame_files):
#         window = frame_files[i:i + n_frames]
#         sequence = [os.path.join(frames_path, f) for f in window]
#         all_sequences.append(sequence)
#         i += stride
#     if len(all_sequences[-1]) < 3:
#         all_sequences = all_sequences[:-1]
#     return all_sequences


def load_frame_sequences(frames_path, n_frames=6, stride=5):

    if stride >= n_frames:
        raise ValueError(f"stride ({stride}) must be less than n_frames ({n_frames}).")

    frame_files = sorted(
        [f for f in os.listdir(frames_path) if f.lower().endswith('.jpg')],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    all_sequences = []
    i = 0
    N = len(frame_files)

    while i < N:
        end = i + n_frames
        window_files = frame_files[i:end]
        if not window_files:
            break
        sequence = [os.path.join(frames_path, f) for f in window_files]
        all_sequences.append(sequence)
        if end >= N:
            break
        i += stride

    return all_sequences


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
                                    temperature=temperature,
                                    top_p=top_p,
                                    pad_token_id=model.config.eos_token_id)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded

def extract_accident_label(llm_output: str):
    """
    Extract accident detection label from LLM output.
    Handles both single and double quotes.
    If not found, defaults to: {'Accident': 'No', 'Frame': 'N/A'}
    """
    pattern = r"""\[\s*['"]Accident['"]\s*:\s*['"](Yes|No)['"]\s*,\s*['"]Frame['"]\s*:\s*(['"]?(N/A|\d+)['"]?)\s*\]"""
    match = re.search(pattern, llm_output, re.IGNORECASE)

    if match:
        accident = match.group(1).capitalize()
        frame_raw = match.group(2).strip("'\"")

        if frame_raw.upper() == 'N/A':
            frame = 'N/A'
        else:
            frame = int(frame_raw)

        return {'Accident': accident, 'Frame': frame}
    else:
        return {'Accident': 'No', 'Frame': 'N/A'}

def evaluation(dataset, model, processor, args):
    counter = 0
    acc_counter = 0
    frame_diff = 0
    start_time = time.time()
    true_target = []
    predict_target = []
    correct = 0
    case_num = len(dataset)

    # Create result folder
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = str(args.model_path.split('-')[2])
    result_folder = f"./Experiment Results Log/{model_name}_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    args_dict = vars(args)
    args_save_path = os.path.join(folder_path, "configs.json")
    with open(args_save_path, 'w') as f:
        json.dump(args_dict, f, indent=4)


    for instance in dataset:
        counter += 1
        video_path = instance[0] # '/.../SO_TAD/test/404.mp4'
        frames_path = instance[1] # '/.../SO_TAD/Extracted_frames/404'
        accident_label = instance[2] # 0 or 1
        true_target.append(accident_label) # 0 or 1
        init_frame = int(instance[3]) # -1 or index

        print('=========================================')
        print('Work on video: ', video_path)
        image_list = load_frame_sequences(frames_path = frames_path, n_frames=args.n_frames, stride=args.stride)
        print('Extracted clips No.: ', len(image_list))

        clip_init = 0

        for clip_frames in image_list:

            messages = message_template(clip_frames)
            # inference
            model_output = inference(processor, model, messages, args.max_new_tokens, args.temperature, args.top_p)

            print('-----------------------------------------')
            print('LLM outputs using frames: ')
            print(model_output)
            print('-----------------------------------------')

            llm_fmt_res = extract_accident_label(model_output)

            print('Extracted results: ')
            print(llm_fmt_res)
            if llm_fmt_res['Accident'] == 'Yes':
                flag = 1
                if llm_fmt_res['Frame'] == 'N/A':
                    frame_idx = 0 # --> assume took place at the first frame
                    break # stop for
                else:
                    frame_idx = int(llm_fmt_res['Frame']) + clip_init
                    break # stop for
            else: # -> predict as no accident
                flag = 0
                clip_init = clip_init + args.stride
                if accident_label == 1:
                    frame_idx = init_frame + 250

        predict_target.append(flag)

        print('-----------------------------------------')
        if accident_label == 1:
            print('True label: Yes')
            # calculate NRMSE
            acc_counter += 1 # accident counter
            frame_diff = (frame_idx - init_frame) ** 2 + frame_diff
        else:
            print('True label: No')

        # calculate statistics & save
        if counter % 10 == 0:
            # save results
            res_dic = {'true_target':true_target, 'predict_target':predict_target}
            save_path = os.path.join(folder_path, f"log_after_{counter}_videos.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(res_dic, f)
            print(f"Saved experiment data at step {counter} to {save_path}")

        if flag == accident_label:
            correct += 1
            print(f"Case {video_path.split('/')[-1]} correctly predicted!")
        else:
            print(f"Case {video_path.split('/')[-1]} wrongly predicted!")

    end_time = time.time()
    print('Video-level video classification accuracy: ', correct/case_num)
    auc = roc_auc_score(true_target, predict_target)
    print(f"AUC: {auc:.4f}")
    print('Confusion Matrix: ')
    print(sklearn.metrics.confusion_matrix(true_target, predict_target, labels=[0, 1]))
    print(sklearn.metrics.classification_report(true_target, predict_target, zero_division=0))
    duration_time = (end_time - start_time)/case_num
    print(f'Average inference time for a case: {duration_time:.2} s')

    # canculate NRMSE
    rmse = (frame_diff / acc_counter) ** 0.5
    nrmse = rmse/250
    print('NRMSE: ', nrmse)

def main():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--model_path',
                        default='./Models/llava-v1.6-vicuna-7b-hf',
                        type=str,
                        help='Local path of your pretrained model')
    parser.add_argument('--max_new_tokens',
                        default=512,
                        help='The maximum number of new tokens generated (i.e. the maximum length of the model output)')
    parser.add_argument('--temperature',
                        default=0.2,
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

    # Dataset parameters
    parser.add_argument('--test',
                        default='./Benchmark/SO_TAD/test',
                        type=str)
    parser.add_argument('--labels',
                        default='./Benchmark/SO_TAD/Appendix.txt',
                        type=str)
    parser.add_argument('--output_frames',
                        default='./Benchmark/SO_TAD/Extracted_frames',
                        type=str)

    args = parser.parse_args()
    # prepare data
    labels = load_labels(args.labels)
    testing_set = extract_labels(args.test, args.output_frames, labels)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
        quantization_config=bnb_config
    ).to('cuda').eval()
    processor = LlavaNextProcessor.from_pretrained(args.model_path, use_fast=True)

    # Evaluate
    # Only report performance on the testing data
    print('--------Evaluate on testing set now!----------')
    evaluation(testing_set, model, processor, args)

if __name__ == '__main__':
    main()