import argparse
import sklearn.metrics
from sklearn.metrics import roc_auc_score
import time
import os
import re
import pickle
from datetime import datetime
from dashscope import MultiModalConversation
import json

def extract_accident_label(llm_output: str,args):
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
            frame = upsample_frame_index(int(frame_raw),args.fps)

        return {'Accident': accident, 'Frame': frame}
    else:
        return {'Accident': 'No', 'Frame': 'N/A'}

def message_template(video_path, fps):
    system_input = """You are a helpful and intelligent assistant. You will be provided with a traffic surveillance video captured by a roadside camera.
    Your task is to analyze the video based on the user's request.
    Specifically, determine whether a traffic accident occurs in the video. If an accident is detected, also identify and report the index of the frame where the accident begins.
    You must include accident prediction results at the end in a specific format: ['Accident': 'Yes' or 'No', 'Frame': 'N/A' or index]"""
    user_input_1 = """Here's an example for you~
    Q: 
    This is a traffic video captured by a road surveillance camera. Please analyze it step by step from the following three perspectives:
    1. **Road Network**: Describe the road structure, such as intersections, straight roads, curves, number of lanes, traffic signs, or lights.
    2. **Traffic Participants**: Identify the types of traffic participants present (e.g., sedans, buses, motorcycles, pedestrians).
    3. **Motion Behaviors**: Describe the movement patterns of the traffic participants (e.g., speed, direction, lane changes, sudden stops, rule violations).
    Finally, based on the motion behaviors and interactions among the traffic participants, determine whether a traffic accident occurred. If an accident is detected, specify the **index of the first frame where the accident happens**.
    Present your answer in a structured format, and conclude with a summary result in this format:  
    `['Accident': 'Yes' or 'No', 'Frame': 'N/A' or index]`
    Let's think step by step.
    ----------
    A:
    Firstly, let's check out the road network~
    ## Road Network
    The video shows a multi-lane urban road with the following features:
    1. Intersection: The road appears to be approaching an intersection, as indicated by traffic lights and signage (e.g., a "No Entry" sign on the left and speed limit signs).
    2. Lanes: There are multiple lanes for vehicles, including at least two lanes for cars and possibly a dedicated lane for buses or other large vehicles.
    Next, I need to identify the traffic participants and their behaviors~
    ## Traffic Participants
    1. Buses: A prominent blue bus is seen traveling in the center lane.
    2. Cars: Several sedans and smaller passenger vehicles are visible, moving in both directions.
    3. Motorcycles/Scooters: Multiple motorcycles and scooters are seen, particularly on the right side of the road.
    ## Motion Behaviors of Traffic Participants
    1. Bus: The blue bus is moving steadily forward in the center lane, maintaining a consistent speed.
    2. Cars: Cars are driving in both directions, with some accelerating and others maintaining a steady pace. Vehicles appear to be adhering to traffic signals and lane discipline.
    3. Motorcycles/Scooters: Motorcycles and scooters are seen moving quickly, often weaving between lanes or staying close to the edge of the road. A motorcycle sped across the intersection from the right to the left.
    Lastly, based on the observation, i need to find out whether a traffic accident occurred in the video.
    ## Accident detection
    A motorcycle was crossing the intersection from the right to the left at high speed. After avoiding the blue bus, the motorcycle collided with the black car. The first frame where the accident occurred was frame 45.
    ## detection results
    ['Accident': 'Yes', 'Frame': 45]"""
    video_path_1 = "./Benchmark/SO_TAD/train/176.mp4"
    ass_input_1 = """Understood. I will proceed step by step:
    ### Step 1: Road Network  
    I will observe the road structure, such as whether it’s an intersection or straight road, the number of lanes, and the presence of traffic signs or signals.
    ### Step 2: Traffic Participants  
    Then I will identify the vehicles or pedestrians present in the video, such as cars, buses, motorcycles, or bicycles.
    ### Step 3: Motion Behaviors  
    Next, I will analyze how each traffic participant moves — their direction, speed, lane changes, or any irregular behaviors.
    ### Step 4: Accident Judgment  
    Based on the above observations, I will determine whether any traffic accident occurs. If so, I will locate the **frame index** where the first collision or abnormal interaction begins.
    ### Final Output  
    I will summarize the accident detection result using the required format.  
    For example: `['Accident': 'Yes', 'Frame': 45]`
    """
    user_input_2 = """Describe this video from the three perspectives and determine whether there is an accident and the initial frame of the accident."""
    video_path_2 = "./Benchmark/SO_TAD/train/623.mp4"
    ass_input_2 = """
    ## Step 1: Road Network
    The video shows a multi-lane highway with multiple lanes in each direction, separated by a central divider.  
    - There are at least three lanes visible in each direction, allowing for smooth traffic flow and overtaking maneuvers.  
    - The road appears to be relatively straight, with no sharp curves or intersections visible in the observed frames.  
    ## Step 2: Traffic Participants
    The highway is populated with a variety of vehicles:  
    - Predominantly cars and trucks, with possible sightings of buses.  
    - Vehicles are distributed across all lanes, moving steadily in both directions.
    ## Step 3: Motion Behaviors of Traffic Participants
    Observations about the motion behavior include:  
    - Vehicles are moving at consistent speeds and in an orderly manner.  
    - Occasional lane changes are seen, particularly in the leftmost lanes, which are likely used for overtaking.  
    - No signs of sudden stops, swerving, or dangerous maneuvers are observed.
    ## Step 4: Accident Detection
    Based on the above observations:  
    - There is no visible evidence of a traffic accident.  
    - All vehicles are moving predictably, and no collisions or abnormal behaviors (e.g., braking, drifting, or vehicle damage) are detected.  
    - Therefore, no specific frame indicating the start of an accident can be identified.
    ## Final Result
    ['Accident': 'No', 'Frame': 'N/A']
    """
    user_input_3 = """Describe this new given video from the three perspectives and determine whether there is an accident and the initial frame of the accident."""
    messages = [
        {"role": "system", "content": system_input},
        {"role": "user", "content": [
            {"text": user_input_1},
            {"video": video_path_1, 'fps': fps},
        ]
         },
        {"role": "assistant",
         "content": ass_input_1
         },
        {"role": "user", "content": [
            {"text": user_input_2},
            {"video": video_path_2, 'fps': fps},
        ]
         },
        {"role": "assistant",
         "content": ass_input_2
         },
        {"role": "user", "content": [
            {"text": user_input_3},
            {"video": video_path, 'fps': fps},
        ]
         }
    ]
    return messages

def detect_accident(llm_output):
    text = llm_output.lower().strip()

    # negate accidents (priority judgment)
    negative_patterns = [
        r'\bno (accident|incident|collision|crash)\b',
        r'does not.*(show|contain|indicate).*(accident|incident|collision|crash)',
        r'without.*(accident|incident|collision|crash)',
        r'no signs of.*(accident|incident)',
        r'no evidence of.*(accident|incident)',
    ]

    for pat in negative_patterns:
        if re.search(pat, text):
            return 0

    # accidents
    positive_patterns = [
        r'\b(accident|incident|collision|crash)\b',
        r'(vehicle|car|bus).*hit.*(another|pedestrian|object)',
        r'(involved|engaged).*in.*(accident|incident)',
    ]

    for pat in positive_patterns:
        if re.search(pat, text):
            return 1

    return 0 #Unclear

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

def upsample_frame_index(index_model_fps, model_fps):
    return int(round(index_model_fps * 25 / model_fps))

def evaluation(dataset, args):
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

    model_name = str(args.model_name.split('/')[-1])
    result_folder = f"./Experiment Results Log/{model_name}_{current_time}"

    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    args_dict = vars(args)
    args_save_path = os.path.join(folder_path, "configs.json")
    with open(args_save_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

    for instance in dataset:
        counter += 1
        video_path = instance[0]
        # frames_path = instance[1]
        accident_label = instance[2]
        true_target.append(accident_label) # 0 or 1
        init_frame = int(instance[3])

        # prepare message
        messages = message_template(video_path, args.fps)

        # inference
        response = MultiModalConversation.call(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model=args.model_name,
            stream=False,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=42,
            messages=messages)

        time.sleep(2)

        model_output = response["output"]["choices"][0]["message"].content[0]["text"]
        print('=========================================')
        print(model_output)
        print('=========================================')

        llm_fmt_res = extract_accident_label(model_output,args)
        print('Extracted results: ')
        print(llm_fmt_res)
        print('=========================================')


        if llm_fmt_res['Accident'] == 'Yes':
            flag = 1
            if llm_fmt_res['Frame'] == 'N/A':
                frame_idx = 0 # --> assume took place at the first frame
            else:
                frame_idx = int(llm_fmt_res['Frame'])
        else:
            flag = 0
            if accident_label == 1:
                frame_idx = init_frame + 250
        predict_target.append(flag)

        if accident_label == 1:
            print('True label: Yes')
            # calculate NRMSE
            acc_counter += 1
            frame_diff = (frame_idx - init_frame) ** 2 + frame_diff
        else:
            print('True label: No')

        if counter % 10 == 0:
            # save results
            res_dic = {'true_target':true_target, 'predict_target':predict_target}
            save_path = os.path.join(folder_path, f"exp_data_{counter}.pkl")
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
    duration_time = (end_time - start_time) / case_num
    print(f'Average inference time for a case: {duration_time:.2} s')
    # canculate NRMSE
    rmse = (frame_diff / acc_counter) ** 0.5
    nrmse = rmse/250
    print('NRMSE: ', nrmse)



def main():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--model_name',
                        default='qwen2.5-vl-32b-instruct',
                        type=str,
                        help='Model Name')
    parser.add_argument('--max_tokens',
                        default=512,
                        help='The maximum number of new tokens generated (i.e. the maximum length of the model output)')
    parser.add_argument('--fps',
                        default=10,
                        help='Video FPS')
    parser.add_argument('--temperature',
                        default=0.2,
                        help='The smaller the parameter value, the more certain the model will return.')
    parser.add_argument('--top_p',
                        default=0.95,
                        help='Nucleus sampling, the smaller the parameter value, the more certain the model will return.')

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

    # Evaluate
    # evaluation(training_set, model, processor, args)
    # Only report performance on the testing data
    print('--------Evaluate on testing set now!----------')
    evaluation(testing_set, args)

if __name__ == '__main__':
    main()