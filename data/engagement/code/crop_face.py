import os
import cv2
import sys

from PIL import Image
from tqdm import tqdm
from utils import findlandmark, image_align

video_list = []

for video_name in sorted(os.listdir("../video")):
    for clip1 in tqdm(sorted(os.listdir(f"../video/{video_name}"))):
        for clip2 in sorted(os.listdir(f"../video/{video_name}/{clip1}")):
            video_list.append([video_name, clip1, clip2])

for video_sample in tqdm(video_list):
    video_name, clip1, clip2 = video_sample
    video_path = f"../video/{video_name}/{clip1}/{clip2}"
    frame_path = f"../frame/{video_name}/{clip1}/{clip2[:-4]}"
    os.makedirs(frame_path, exist_ok=True)
    aligned_frame_path = f"../aligned_frame/{video_name}/{clip1}/{clip2[:-4]}"
    os.makedirs(aligned_frame_path, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % 5 == 0:
            img_path = f"{frame_path}/{count:04d}.png"
            cv2.imwrite(img_path, image)
        success, image = vidcap.read()
        count += 1

    for image_name in sorted(os.listdir(frame_path)):
        img_path = f"{frame_path}/{image_name}"
        aligned_img_path = f"{aligned_frame_path}/{image_name}"
        if not os.path.exists(aligned_img_path):
            landmark, success = findlandmark(img_path)
            if success == True:
                aligned_image = image_align(Image.open(img_path), landmark)
                aligned_image.save(aligned_img_path)
