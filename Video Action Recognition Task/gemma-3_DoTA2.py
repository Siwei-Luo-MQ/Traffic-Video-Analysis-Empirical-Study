import argparse
import json
import os
from datetime import datetime
import numpy as np
from pathlib import Path
import torch
import re
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

def message_template(sampled_images,args):
    with open(args.system_prompts, "r", encoding="utf-8") as f:
        sys_input = f.read()
    user_input_3 = "Please think step by step, analyze the provided frames, classify any abnormal traffic behavior shown in the video, and finally output the result in the specified format."
    content_list = [{"type": "image", "image": str(img)} for img in sampled_images]
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

def extract_prediction_result(response_text):
    pattern = r"\['Abnormal Event':\s*'(?P<event>[A-Z/]+|N/A)'\s*,\s*'Ego Involved':\s*'(?P<ego>Yes|No)'\s*\]"
    match = re.search(pattern, response_text)

    if match:
        return {
            "Abnormal Event": match.group("event"),
            "Ego Involved": match.group("ego")
        }
    else:
        return {
            "Abnormal Event": "Unknown",
            "Ego Involved": "Unknown"
        }
    
def run_gemma(sampled_images, args, model, processor):
    messages = message_template(sampled_images,args)
    # inference
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
                                    max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature,
                                    top_p=args.top_p)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded

def acc_class(correct_per_class, case_no_per_class):
    accuracy_per_class = np.zeros(16)
    valid_mask = (case_no_per_class != 0) & (correct_per_class != 0)
    accuracy_per_class[valid_mask] = correct_per_class[valid_mask] / case_no_per_class[valid_mask]
    return accuracy_per_class

def print_formatted_acc_per_class(acc_array):
    if len(acc_array) != 16:
        raise ValueError("Accuracy array must have 16 elements.")

    categories = [
        'ego - ST', 'ego - AH', 'ego - LA', 'ego - OC',
        'ego - TC', 'ego - VP', 'ego - VO', 'ego - OO',
        'other - ST', 'other - AH', 'other - LA', 'other - OC',
        'other - TC', 'other - VP', 'other - VO', 'other - OO'
    ]

    print("=" * 40)
    print("Per-Class Accuracy:")
    print("-" * 40)

    for i in range(8):
        print(f"{categories[i]:<15}: {acc_array[i]:.2%}")
    print("-" * 40)
    for i in range(8, 16):
        print(f"{categories[i]:<15}: {acc_array[i]:.2%}")

    print("=" * 40)

def video_encoding(data_folder, args, video_metadata=None):
    data_folder = Path(data_folder)

    image_paths = sorted([p for p in data_folder.glob("*.jpg") if p.is_file()])
    total_images = len(image_paths)

    # 如果设置了使用anomaly片段
    if args.use_anomaly_segment_only and video_metadata is not None:
        start_idx = int(video_metadata.get("anomaly_start", 0))
        end_idx = int(video_metadata.get("anomaly_end", total_images - 1))
        anomaly_indices = list(range(start_idx, end_idx + 1))
        anomaly_indices = [i for i in anomaly_indices if i < total_images]
        if len(anomaly_indices) <= args.sampled_num_frames:
            sampled_indices = anomaly_indices
        else:
            sampled_indices = [
                anomaly_indices[int(round(i * (len(anomaly_indices) - 1) / (args.sampled_num_frames - 1)))]
                for i in range(args.sampled_num_frames)
            ]
        sampled_images = [image_paths[i] for i in sampled_indices]
    else:
        # 原有采样策略
        if total_images <= args.sampled_num_frames:
            sampled_images = image_paths
        else:
            indices = [
                int(round(i * (total_images - 1) / (args.sampled_num_frames - 1)))
                for i in range(args.sampled_num_frames)
            ]
            sampled_images = [image_paths[i] for i in indices]

    # encoded_images = []
    # for image_path in sampled_images:
    #     with open(image_path, 'rb') as f:
    #         encoded = base64.b64encode(f.read()).decode('ascii')
    #         encoded_images.append(encoded)
    return sampled_images

def count_jpg_files(folder_path):
    return sum(
        1 for file in os.listdir(folder_path)
        if file.lower().endswith('.jpg')
    )

def label_transform(anomaly_class):
    anomaly_class_dict = {
        'ego: start_stop_or_stationary': {'Abnormal Event': 'ST', 'Ego Involved': 'Yes'},
        'ego: moving_ahead_or_waiting': {'Abnormal Event': 'AH', 'Ego Involved': 'Yes'},
        'ego: lateral': {'Abnormal Event': 'LA', 'Ego Involved': 'Yes'},
        'ego: oncoming': {'Abnormal Event': 'OC', 'Ego Involved': 'Yes'},
        'ego: turning': {'Abnormal Event': 'TC', 'Ego Involved': 'Yes'},
        'ego: pedestrian': {'Abnormal Event': 'VP', 'Ego Involved': 'Yes'},
        'ego: obstacle': {'Abnormal Event': 'VO', 'Ego Involved': 'Yes'},
        'ego: leave_to_right': {'Abnormal Event': 'OO', 'Ego Involved': 'Yes'},
        'ego: leave_to_left': {'Abnormal Event': 'OO', 'Ego Involved': 'Yes'},
        'other: start_stop_or_stationary': {'Abnormal Event': 'ST', 'Ego Involved': 'No'},
        'other: moving_ahead_or_waiting': {'Abnormal Event': 'AH', 'Ego Involved': 'No'},
        'other: lateral': {'Abnormal Event': 'LA', 'Ego Involved': 'No'},
        'other: oncoming': {'Abnormal Event': 'OC', 'Ego Involved': 'No'},
        'other: turning': {'Abnormal Event': 'TC', 'Ego Involved': 'No'},
        'other: pedestrian': {'Abnormal Event': 'VP', 'Ego Involved': 'No'},
        'other: obstacle': {'Abnormal Event': 'VO', 'Ego Involved': 'No'},
        'other: leave_to_right': {'Abnormal Event': 'OO', 'Ego Involved': 'No'},
        'other: leave_to_left': {'Abnormal Event': 'OO', 'Ego Involved': 'No'},
    }
    return anomaly_class_dict[anomaly_class]

def label_decoder(label):
    if label['Abnormal Event'] == 'ST' and label['Ego Involved'] == 'Yes':
        index_n = 0
    elif label['Abnormal Event'] == 'AH' and label['Ego Involved'] == 'Yes':
        index_n = 1
    elif label['Abnormal Event'] == 'LA' and label['Ego Involved'] == 'Yes':
        index_n = 2
    elif label['Abnormal Event'] == 'OC' and label['Ego Involved'] == 'Yes':
        index_n = 3
    elif label['Abnormal Event'] == 'TC' and label['Ego Involved'] == 'Yes':
        index_n = 4
    elif label['Abnormal Event'] == 'VP' and label['Ego Involved'] == 'Yes':
        index_n = 5
    elif label['Abnormal Event'] == 'VO' and label['Ego Involved'] == 'Yes':
        index_n = 6
    elif label['Abnormal Event'] == 'OO' and label['Ego Involved'] == 'Yes':
        index_n = 7
    elif label['Abnormal Event'] == 'ST' and label['Ego Involved'] == 'No':
        index_n = 8
    elif label['Abnormal Event'] == 'AH' and label['Ego Involved'] == 'No':
        index_n = 9
    elif label['Abnormal Event'] == 'LA' and label['Ego Involved'] == 'No':
        index_n = 10
    elif label['Abnormal Event'] == 'OC' and label['Ego Involved'] == 'No':
        index_n = 11
    elif label['Abnormal Event'] == 'TC' and label['Ego Involved'] == 'No':
        index_n = 12
    elif label['Abnormal Event'] == 'VP' and label['Ego Involved'] == 'No':
        index_n = 13
    elif label['Abnormal Event'] == 'VO' and label['Ego Involved'] == 'No':
        index_n = 14
    elif label['Abnormal Event'] == 'OO' and label['Ego Involved'] == 'No':
        index_n = 15
    return index_n

def filter_unknown_anomalies(json_path, save_path=None):
    with open(json_path, 'r') as f:
        data = json.load(f)

    filtered_data = {
        k: v for k, v in data.items()
        if v.get("anomaly_class") not in ["other: unknown", "ego: unknown"]
    }

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)

    return filtered_data

def main():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--model_path',
                        default='./Models/gemma-3-12b-it',
                        type=str,
                        help='Local path of your pretrained model')
    parser.add_argument('--max_new_tokens',
                        default=512,
                        help='The maximum number of new tokens generated (i.e. the maximum length of the model output)')
    parser.add_argument('--temperature',
                        default=0.4,
                        help='The smaller the parameter value, the more certain the model will return.')
    parser.add_argument('--top_p',
                        default=0.95,
                        help='Nucleus sampling, the smaller the parameter value, the more certain the model will return.')
    parser.add_argument('--sampled_num_frames',
                        default=25,
                        type=int,
                        help='No. frames sampled from videos')
    # Dataset parameters
    parser.add_argument('--use_anomaly_segment_only',
                        action='store_true',
                        help='If set, only frames between anomaly_start and anomaly_end are sampled')
    parser.add_argument('--val',
                        default='./Benchmark/DoTA/metadata_val.json',
                        type=str)
    parser.add_argument('--data_folder',
                        default='./Benchmark/DoTA/validation/',
                        type=str)
    parser.add_argument('--system_prompts',
                        default='./system.txt',
                        type=str)
    args = parser.parse_args()
    validation = filter_unknown_anomalies(args.val)

    print('=========================Smart Traffic Project========================')
    print('============== detecting traffic anomalies on DoTA dataset============')
    print('Experiment Settings:')
    print('Foundation Model - Gemma-3-12b-it')
    print(f'temperature: {args.temperature}, top_p: {args.top_p},  number of sampled frames: {args.sampled_num_frames}')
    print(f'No. videos: {len(validation)}')
    if args.use_anomaly_segment_only:
        print(f"Sampling anomaly segment only.")
    else:
        print("Sampling from full video.")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)

    # Create result folder
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"./Experiment_results/Gemma_3_12b_DoTA_results_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    # saving cases with ContentPolicyViolation
    # policy_violation_log_path = os.path.join(folder_path, "content_policy_violations.jsonl")

    data_size = len(validation)

    # metrics needed to calculate
    # ACG = No. correctly classified cases / data_size
    correct_num = 0
    # Acc under each class, 16 classes in total
    case_no_per_class = np.zeros(16)
    correct_per_class = np.zeros(16)

    # effective_data_size = 0  # Total number of cases without filtering
    # effective_case_no_per_class = np.zeros(16)
    # effective_correct_per_class = np.zeros(16)

    num_cases_till_now = 1

    for key, value in validation.items():
        data_id = key
        anomaly = value.get("anomaly_class")
        # decoded label {'Abnormal Event': 'OO', 'Ego Involved': 'No'}
        data_label = label_transform(anomaly)
        case_no_per_class[label_decoder(data_label)] = case_no_per_class[label_decoder(data_label)] + 1

        data_folder = args.data_folder + data_id
        num_frames = count_jpg_files(data_folder)
        print(f'Working on the video - {data_id}. Num of frames: {num_frames}, Class - {anomaly}')
        
        sampled_images = video_encoding(data_folder, args, value)

        model_output = run_gemma(sampled_images, args, model, processor)
        print('-----------------------------------------')
        print('LLM outputs: ')
        print(model_output)
        print('-----------------------------------------')
        prediction = extract_prediction_result(model_output)
        print('====+++====')
        print(f'Predicted label: {prediction}')
        print('-----+++++-----')
        print("True Label: ",data_label)

        is_correct = prediction['Abnormal Event'] == data_label['Abnormal Event'] and \
                     prediction['Ego Involved'] == data_label['Ego Involved']

        if is_correct:
            correct_per_class[label_decoder(data_label)] += 1
            correct_num += 1
            print('+++++++++++++++++++++++++++')
            print('Correctly predicted! Move to the next video ...')
        else:
            print('+++++++++++++++++++++++++++')
            print('Wrongly predicted! Move to the next video ...')
        
        if num_cases_till_now % 20 == 0:
            print(f"-----Saving results after {num_cases_till_now} cases...-----")
            current_acg = correct_num / num_cases_till_now
            current_acc_class = acc_class(correct_per_class, case_no_per_class)
            results_dict = {
                "current_acc_overall": current_acg,
                "current_acc_per_class": current_acc_class.tolist()
            }
            print('Current overall accuracy: ', current_acg)
            print_formatted_acc_per_class(current_acc_class)

            save_path = os.path.join(folder_path, f'results_after_{num_cases_till_now}_cases.json')
            with open(save_path, 'w') as f:
                json.dump(results_dict, f, indent=4)

            print(f"Saved metrics to: {save_path}")

        num_cases_till_now += 1
    # all videos are checked
    # calculate ACG
    print('================== Final Evaluation ==================')
    acg = correct_num / data_size
    print('Accuracy: ', acg)
    print_formatted_acc_per_class(acc_class(correct_per_class, case_no_per_class))



if __name__ == '__main__':
    main()