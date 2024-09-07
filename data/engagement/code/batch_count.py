import os
import csv
import random
import numpy as np
import pandas as pd
import krippendorff

from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score


def convert(ans):
    if ans == "Never":
        return 1
    elif ans == "Rarely":
        return 2
    elif ans == "Sometimes":
        return 3
    elif ans == "Often":
        return 4
    elif ans == "Very often":
        return 5
    elif ans == "No":
        return 1
    elif ans == "Unsure":
        return 3
    elif ans == "Yes":
        return 5

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--fold_id", type=int, default=0)
    parser.add_argument("--targetmode", type=str, default="reg")
    parser.add_argument("--filter_mode", type=bool, default=False)
    args = parser.parse_args()
    cur_fold_id = args.fold_id
    targetmode = args.targetmode
    filter_mode = args.filter_mode
    batch_count = {}
    annotation_dict = {}
    engagement_min, engagement_max = -0.4, 0.4
    th1, th2 = engagement_min + (engagement_max - engagement_min) / 3, engagement_min+ (engagement_max - engagement_min) * 2 / 3
    reg_th = 1000
    for date in ["0326", "0327_1", "0327_2"]:
        data_frame = pd.read_csv(f"main_{date}.csv")
        for index, row in data_frame.iterrows():
            if index == 0 or index == 1:
                continue # skip title
            # load annotation
            prolific_id = row["Annotator Name"]
            annotation = []
            batch = row[f"Video1"].split("/")[5]
            if not batch in batch_count:
                batch_count[batch] = 0
            batch_count[batch] += 1

            for video in range(1, 20):
                engagement = 0
                if str(row[f"Video{video}"]) !="nan":
                    clip_name = row[f"Video{video}"].split("/")[6]
                else:
                    continue
                if row[f"{video}_Speaking Question"] != "No":
                    print(f"{batch}_{clip_name}")
                    print(row[f"{video}_Speaking Question"])
                    import sys
                    continue
                if row[f"Video{video}"] != row[f"Video{video}"]:
                    continue
                
                clip_name = row[f"Video{video}"].split("/")[6]
                for question in range(1, 7):
                    if row[f"{video}_Q{question}"] != row[f"{video}_Q{question}"]: # check nan value
                        engagement = None
                        break
                    else:
                        if question in [1, 2, 3, 6]:
                            engagement += (convert(row[f"{video}_Q{question}"]) - 3) / (5*6)
                        elif question in [4, 5]:
                            engagement += (6-convert(row[f"{video}_Q{question}"]) - 3) / (5*6)
                if engagement == None:
                    continue
                if f"{batch}_{clip_name}" not in annotation_dict:
                    annotation_dict[f"{batch}_{clip_name}"] = []
                if targetmode == 'reg':
                    engagement = engagement
                elif targetmode == '2clf':
                    engagement = (engagement >= 0)# to int
                    engagement = int(engagement)
                else:
                    if engagement <= th1:
                        engagement = 0
                    elif engagement <= th2:
                        engagement = 1
                    else:
                        engagement = 2

                annotation_dict[f"{batch}_{clip_name}"].append(engagement)

    clip_list = list(annotation_dict.keys())

    new_annotation_dict = {}
    for clip in clip_list:
        annotation = annotation_dict[clip]
        annotation = [x for x in annotation if x is not None]
        annotation = np.array(annotation)
        if len(annotation) <= 1:
            continue
        skip_disagree = (np.max(annotation) - np.min(annotation) > 1 and ('clf' in targetmode)) or ((np.max(annotation) - np.min(annotation) > reg_th) and targetmode == 'reg')

        if skip_disagree:
            continue
        new_annotation_dict[clip] = annotation[:3].tolist()
        random.shuffle(new_annotation_dict[clip])

        if len(annotation) == 2:
            new_annotation_dict[clip].append(np.nan)

    new_clip_list = list(new_annotation_dict.keys())
    new_annotation_list = []
    for clip in new_clip_list:
        new_annotation_list.append(new_annotation_dict[clip])
    new_annotation_list = np.array(new_annotation_list)
    
    print(len(new_clip_list)) # 4674
    print(new_annotation_list.shape) # 4674, 3
    if not targetmode == 'reg':
        kappa = cohen_kappa_score(new_annotation_list[:, 0], new_annotation_list[:, 1])
        print(kappa) # 0.412
    alpha = krippendorff.alpha(reliability_data=[new_annotation_list[:, 0], new_annotation_list[:, 1]], level_of_measurement="interval")
    print(alpha) # 0.549

    annotations = []
    clips = []
    for i, annotation in enumerate(new_annotation_list):

        if targetmode == 'reg':
            nonskip_disagree = (not np.isnan(annotation[2])) and (annotation[0] - annotation[2] <= 10) 
        else:
            nonskip_disagree = (not np.isnan(annotation[2])) and (annotation[0] - annotation[2] <= 1)
        if nonskip_disagree:
            annotations.append(annotation)
            clips.append(new_clip_list[i])
        
    annotations = np.array(annotations)
    print(len(clips)) # 3476
    print(annotations.shape) # 3476, 3
    if not targetmode == 'reg':
        kappa_scores = []
        for i in range(2):
            for j in range(i+1, 3):
                kappa = cohen_kappa_score(annotations[:, i], annotations[:, j])
                print(kappa)
                kappa_scores.append(kappa)

        average_kappa = np.mean(kappa_scores)
        print(f"Cohen Kappa: {average_kappa}") # 0.407
    alpha = krippendorff.alpha(reliability_data=[annotations[:, 0], annotations[:, 1], annotations[:, 2]], level_of_measurement="interval")
    print(alpha) # 0.545
    
    import sys

    all_videos = [
        "Ask_a_Therapist_How_to_Manage_Mental_Health_During_a_Pandemic",
        "August_Facebook_Live",
        "Dry_Eye_Zoom_Group_October_14_2022",
        "Early_Mid_StageCare_Support_Group_Webinar_10th_September_2021",
        "Group_therapy_video_2",
        "Mock_Group_Therapy_Session_Substance_Abuse",
        "PCA_Support_Group_Webina-9th_December_2022",
        "PCA_Support_Group_Webinar_-_1st_December_2023",
        "PTSD_Buddies_Zoom_Group_Support_Meeting",
        "Stroke_Buddies_Support_Group_Meeting_#5",
         "Te_Awamutu_Community_Board_Meeting_14_September_2021",
        "Zoom_Focus_Group",
        "grief_support_group",
        "zoom_group_therapy_session_1",
    ]
    
    num_folds = 5
    kfolds_mode = True
    stratified_mode = False
    if 1 and kfolds_mode:
        fold_size = len(all_videos) // num_folds
        remainder = len(all_videos) % num_folds
        
        val_videos = all_videos[cur_fold_id*fold_size: (cur_fold_id+1)*fold_size]
        train_videos = all_videos[:cur_fold_id*fold_size] + all_videos[(cur_fold_id+1)*fold_size:]

        if stratified_mode:
            val_videos = all_videos[cur_fold_id]
            train_videos = all_videos

            all_vids_fold = [0, 0, 1, 2, 0, 2, 3, 1, 3, 2, 4, 4, 3, 1]
            indices_for_desired_fold = [index for index, fold in enumerate(all_vids_fold) if fold == cur_fold_id]
            val_videos = [all_videos[index] for index in indices_for_desired_fold]
        
            indices_for_desired_fold222 = [i for i in range(len(all_videos)) if i not in indices_for_desired_fold]
            train_videos = [all_videos[index] for index in indices_for_desired_fold222]

       
    if targetmode == 'reg':
        root_path = f"../engagement/Rfinal_4reg_{cur_fold_id}"
    elif targetmode == '2clf':
        root_path = f"../engagement/Rlabel_0402_4CLF2_{cur_fold_id}"
    else:
        root_path = f"../engagement/Rlabel_0402_4CLS_{cur_fold_id}"
    #root_path = f'../engagement/Rsk_0402_{cur_fold_id}'
    os.makedirs(root_path, exist_ok=True)
    title = "video_path,engagement\n"
    train_data, val_data, test_data = [], [], []

    for i, clip in enumerate(new_clip_list):
        clip = clip.split("_")
        for idx, s in enumerate(clip):
            if "clip" in s:
                break
        video_name = "_".join(clip[2:idx])
        clip_name = "_".join(clip[idx+1:])
        engagement = float(np.mean(new_annotation_list[i][:2]))
        if targetmode == '2clf':
            engagement = (engagement >= 1)
            engagement = int(engagement)

        
        frame_path = f"../engagement/aligned_frame/{video_name}/{clip[idx]}/{clip_name[:-4]}"
        audio_path = f"../engagement/audio/{video_name}/{clip[idx]}/{clip_name[:-4]}.wav"
        text_path = f"../engagement/text/{video_name}/{clip[idx]}/{clip_name[:-4]}.txt"

        if len(os.listdir(frame_path)) < 16:
            continue

        if os.path.exists(frame_path) and os.path.exists(audio_path) and os.path.exists(text_path):
            if video_name in train_videos:
                train_data.append([f"{video_name}/{clip[idx]}/{clip_name[:-4]}", engagement])
            elif video_name in val_videos:
                val_data.append([f"{video_name}/{clip[idx]}/{clip_name[:-4]}", engagement])
            else:
                test_data.append([f"{video_name}/{clip[idx]}/{clip_name[:-4]}", engagement])

    train_file = open(f"{root_path}/train.csv", "w")
    train_file.write(title)
    for sample in train_data:
        train_file.write(f"{sample[0]},{sample[1]}\n")

    val_file = open(f"{root_path}/val.csv", "w")
    val_file.write(title)
    for sample in val_data:
        val_file.write(f"{sample[0]},{sample[1]}\n")
    if not kfolds_mode:
        test_file = open(f"{root_path}/test.csv", "w")
        test_file.write(title)
        for sample in test_data:
            test_file.write(f"{sample[0]},{sample[1]}\n")
    if targetmode == 'reg':
        root_path = f"../engagement/Rfinal_3reg_{cur_fold_id}"
    elif targetmode == '2clf':
        root_path = f"../engagement/Rlabel_0402_3CLF2_{cur_fold_id}"
    else:
        root_path = f"../engagement/Rlabel_0402_3CLF_{cur_fold_id}"
    os.makedirs(root_path, exist_ok=True)
    title = "video_path,engagement\n"
    train_data, val_data, test_data = [], [], []

    for i, clip in enumerate(clips):
        clip = clip.split("_")
        for idx, s in enumerate(clip):
            if "clip" in s:
                break
        video_name = "_".join(clip[2:idx])
        clip_name = "_".join(clip[idx+1:])
        engagement = float(np.mean(annotations[i][:2]))
        if targetmode == '2clf':
            engagement = (engagement >= 1.0)
            engagement = int(engagement)

        frame_path = f"../engagement/aligned_frame/{video_name}/{clip[idx]}/{clip_name[:-4]}"
        audio_path = f"../engagement/audio/{video_name}/{clip[idx]}/{clip_name[:-4]}.wav"
        text_path = f"../engagement/text/{video_name}/{clip[idx]}/{clip_name[:-4]}.txt"

        if len(os.listdir(frame_path)) < 16:
            continue

        if os.path.exists(frame_path) and os.path.exists(audio_path) and os.path.exists(text_path):
            if video_name in train_videos:
                train_data.append([f"{video_name}/{clip[idx]}/{clip_name[:-4]}", engagement])
            elif video_name in val_videos:
                val_data.append([f"{video_name}/{clip[idx]}/{clip_name[:-4]}", engagement])
            else:
                test_data.append([f"{video_name}/{clip[idx]}/{clip_name[:-4]}", engagement])

    train_file = open(f"{root_path}/train.csv", "w")
    train_file.write(title)
    for sample in train_data:
        train_file.write(f"{sample[0]},{sample[1]}\n")

    val_file = open(f"{root_path}/val.csv", "w")
    val_file.write(title)
    for sample in val_data:
        val_file.write(f"{sample[0]},{sample[1]}\n")
    if not kfolds_mode:
        test_file = open(f"{root_path}/test.csv", "w")
        test_file.write(title)
        for sample in test_data:
            test_file.write(f"{sample[0]},{sample[1]}\n")
