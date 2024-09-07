import os
import cv2
import shutil
import random
import subprocess
import numpy as np
import moviepy.editor as mp

import torch
import torch.nn as nn

from tqdm import tqdm
from pyannote.audio import Pipeline


class solver_data_process(nn.Module):
    def __init__(self, config):
        super(solver_data_process, self).__init__()
        self.config = config

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",)
        self.pipeline.to(torch.device("cuda"))


    def audio_extraction(self, video_name):
        video_root = f"../data/support_0229/{video_name}/clips"
        for clip1 in sorted(os.listdir(video_root)):
            if clip1 == ".DS_Store":
                continue
            for clip2 in sorted(os.listdir(f"{video_root}/{clip1}")):
                if clip2 == ".DS_Store":
                    continue
                video_path = f"{video_root}/{clip1}/{clip2}"
                audio_path = f"../data/support_0229/{video_name}/audio/{clip1}/{clip2[:-4]}.wav"

                os.makedirs(f"../data/support_0229/{video_name}/audio/{clip1}", exist_ok=True)
                if os.path.exists(audio_path):
                    continue

                if os.path.exists(video_path):
                    command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 16000 -vn {audio_path}"
                    subprocess.call(command, shell=True)


    def audio_segmentation(self, video_name):
        os.makedirs(f"../data/support_0229/{video_name}/segmentation", exist_ok=True)
        audio_root = f"../data/support_0229/{video_name}/audio"

        clip_list = list(os.listdir(audio_root))
        new_clip_list = []
        for clip in clip_list:
            new_clip_list.append(int(clip.replace("clip", "")))
        new_clip_list = sorted(new_clip_list)

        for i in tqdm(new_clip_list):
            clip1 = f"clip{i}"
            for clip2 in sorted(os.listdir(f"{audio_root}/{clip1}")):
                if "clip" in clip2:
                    continue
                if os.path.exists(f"../data/support_0229/{video_name}/segmentation/{clip1}.npy"):
                    continue

                audio_file_name = f"{audio_root}/{clip1}/{clip2}"
                diarization = self.pipeline(audio_file_name)

                split_timestamps = []
                last_start, last_end = 0, 0
                for turn, _, _ in diarization.itertracks(yield_label=True):
                    start, end = turn.start, turn.end
                    if end - last_start > 7.5 and last_end - last_start < 7.5:
                        split_timestamps.append([last_start, end])
                        last_start = end
                        last_end = end
                    else:
                        last_end = end
                if last_end - last_start > 7.5:
                    split_timestamps.append([last_start, last_end])
                split_timestamps = np.array(split_timestamps)
                np.save(f"../data/support_0229/{video_name}/segmentation/{clip1}.npy", split_timestamps)


    def video_segmentation(self, video_name):
        os.makedirs(f"../data/support_segmented_0229/{video_name}/video", exist_ok=True)
        video_root = f"../data/support_0229/{video_name}/clips"

        clip_list = list(os.listdir(video_root))
        new_clip_list = []
        for clip in clip_list:
            new_clip_list.append(int(clip.replace("clip", "")))
        new_clip_list = sorted(new_clip_list)
        segment_idx = 1

        for i in tqdm(new_clip_list):
            clip1 = f"clip{i}"
            split_timestamps = np.load(f"../data/support_0229/{video_name}/segmentation/{clip1}.npy")

            for timestamp in split_timestamps:
                start_time, end_time = timestamp

                clip2 = sorted(os.listdir(f"{video_root}/{clip1}"))[0]
                src_name = f"../data/support_0229/{video_name}/clips/{clip1}/{clip2}"
                duration = self.get_duration(src_name)
                if start_time > duration or end_time > duration:
                    continue
                if end_time - start_time < 7.5 or end_time - start_time > 10:
                    continue

                for clip2 in sorted(os.listdir(f"{video_root}/{clip1}")):
                    src_name = f"../data/support_0229/{video_name}/clips/{clip1}/{clip2}"
                    tgt_name = f"../data/support_segmented_0229/{video_name}/video/clip{segment_idx:04d}/{clip2}"
                    os.makedirs(f"../data/support_segmented_0229/{video_name}/video/clip{segment_idx:04d}", exist_ok=True)

                    my_video = mp.VideoFileClip(src_name)
                    my_clip = my_video.subclip(start_time, end_time)
                    my_clip.write_videofile(tgt_name, logger=None)
                segment_idx += 1


    def get_duration(self, filename):
        video = cv2.VideoCapture(filename)
        fps = video.get(cv2.CAP_PROP_FPS)
        totalNoFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        durationInSeconds = totalNoFrames / fps

        return durationInSeconds


    def video_duration(self, video_name):
        video_root = f"../data/support_segmented_0229/{video_name}/video"
        cnt = 0
        for clip1 in tqdm(sorted(os.listdir(video_root))):
            for clip2 in sorted(os.listdir(f"{video_root}/{clip1}")):
                if not "clip" in clip2:
                    continue
                video_path = f"{video_root}/{clip1}/{clip2}"
                duration = self.get_duration(video_path)

                if duration > 7.5 and duration < 10:
                    cnt += 1

        return cnt


    def video_shuffle(self, video_list):
        video_root = f"../data/support_segmented_0229"
        batch_root = f"../data/batched_support_segmented_0311"

        clip_list = []
        for video in tqdm(video_list):
            for clip1 in sorted(os.listdir(f"{video_root}/{video}/video/")):
                for  clip2 in sorted(os.listdir(f"{video_root}/{video}/video/{clip1}")):
                    if not "clip" in clip2:
                        continue
                    video_path = f"{video_root}/{video}/video/{clip1}/{clip2}"
                    duration = self.get_duration(video_path)

                    if duration > 7.5 and duration < 10:
                        clip_list.append([video, clip1, clip2])

        random.shuffle(clip_list)
        cnt = 0
        for batch_idx in range(10):
            for i in range(32):
                os.makedirs(f"{batch_root}/{batch_idx+1}/{i+1:04d}", exist_ok=True)
                for j in range(20):
                    video, clip1, clip2 = clip_list[batch_idx * 640 + i * 20 + j]
                    src = f"{video_root}/{video}/video/{clip1}/{clip2}"
                    dst = f"{batch_root}/{batch_idx+1}/{i+1:04d}/{video}_{clip1}_{clip2}"
                    shutil.copyfile(src, dst)
                    cnt += 1

        return cnt


    def run(self):
        # video_list = sorted(os.listdir("../data/support_0229"))
        # cnt = 0
        # for video_name in video_list:
        #     print(video_name)
        #     if "Fostering" in video_name:
        #         continue
        #     # self.audio_extraction(video_name)
        #     # self.audio_segmentation(video_name)
        #     # self.video_segmentation(video_name)
        #     cnt += self.video_duration(video_name)

        # print(cnt) # 19004 videos for 25 meetings / 6572 videos for 14 meetings (Fostering Resilience are excluded)

        video_list = []
        for video_name in sorted(os.listdir("../data/support_0229")):
            if "Fostering" in video_name:
                continue
            video_list.append(video_name)
        print(len(video_list))
        cnt = self.video_shuffle(video_list)
        print(cnt) # 3100


# conda create -n pyannote python=3.8
# conda activate pyannote
# conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# pip install librosa
# pip install tqdm
# pip install pyannote.audio
# pip install --upgrade moviepy
# pip install opencv-python

# find files with spaces
# find . -type f -name "* *"
# replace spaces with underscores
# find . -type f -name "* *" | while read file; do mv "$file" ${file// /_}; done
