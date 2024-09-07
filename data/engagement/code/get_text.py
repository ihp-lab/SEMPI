import os
import whisper

from tqdm import tqdm

model = whisper.load_model("tiny.en").cuda()

def get_text(model, audio_path, text_path):
    result = model.transcribe(audio_path)
    text = result["text"]

    transcript_file = open(text_path, "w")
    transcript_file.write(text)
    transcript_file.close()

def get_texts(video_name):
    cnt = 0
    for clip1 in tqdm(sorted(os.listdir(f"../audio/{video_name}"))):
        for clip2 in sorted(os.listdir(f"../audio/{video_name}/{clip1}")):
            if not "clip" in clip2:
                continue
            audio_path = f"../audio/{video_name}/{clip1}/{clip2}"
            text_path = f"../text/{video_name}/{clip1}/{clip2[:-4]}.txt"
            os.makedirs(f"../text/{video_name}/{clip1}", exist_ok=True)

            if not os.path.exists(text_path):
                get_text(model, audio_path, text_path)
            cnt += 1

    return cnt

cnt = 0
for video_name in sorted(os.listdir("../audio")):
    print(video_name)
    cnt += get_texts(video_name)

print(cnt) # 6572 videos for 14 meetings (Fostering Resilience are excluded)
