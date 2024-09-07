import os
import cv2
import shutil

from tqdm import tqdm


def get_duration(filename):
    video = cv2.VideoCapture(filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    totalNoFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    durationInSeconds = totalNoFrames / fps

    return durationInSeconds

def get_videos(video_name):
    video_root = f"../../support_segmented_0229/{video_name}/video"
    cnt = 0
    for clip1 in tqdm(sorted(os.listdir(video_root))):
        for clip2 in sorted(os.listdir(f"{video_root}/{clip1}")):
            if not "clip" in clip2:
                continue
            video_path = f"{video_root}/{clip1}/{clip2}"
            duration = get_duration(video_path)

            if duration > 7.5 and duration < 10:
                os.makedirs(f"../video/{video_name}/{clip1}", exist_ok=True)
                if not os.path.exists(f"../video/{video_name}/{clip1}/{clip2}"):
                    shutil.copy(video_path, f"../video/{video_name}/{clip1}/{clip2}")
                cnt += 1

    return cnt

video_list = sorted(os.listdir("../../support_0229"))
cnt = 0
for video_name in video_list:
    if "Fostering" in video_name:
        continue
    print(video_name)
    cnt += get_videos(video_name)

print(cnt) # 6572 videos for 14 meetings (Fostering Resilience are excluded)
