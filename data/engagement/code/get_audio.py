import os
import cv2
import subprocess

from tqdm import tqdm


def get_audio(video_path, audio_path):
    command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 16000 -vn {audio_path}"
    subprocess.call(command, shell=True)

def get_audios(video_name):
    cnt = 0
    for clip1 in tqdm(sorted(os.listdir(f"../video/{video_name}"))):
        for clip2 in sorted(os.listdir(f"../video/{video_name}/{clip1}")):
            if not "clip" in clip2:
                continue
            video_path = f"../video/{video_name}/{clip1}/{clip2}"
            audio_path = f"../audio/{video_name}/{clip1}/{clip2[:-4]}.wav"
            os.makedirs(f"../audio/{video_name}/{clip1}", exist_ok=True)

            if not os.path.exists(audio_path):
                get_audio(video_path, audio_path)
            cnt += 1

    return cnt

cnt = 0
for video_name in sorted(os.listdir("../video")):
    print(video_name)
    cnt += get_audios(video_name)

print(cnt) # 6572 videos for 14 meetings (Fostering Resilience are excluded)
