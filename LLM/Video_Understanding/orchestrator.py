import os 
from tqdm import tqdm 
from llm_inferencing import model1
import csv 
from pathlib import Path
import re

csv_path = Path("outputs.csv")
model_name = "SmolVLM2-256-Video-Instruct"
video_folder_path = "video_chunks"

if not csv_path.exists():
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Video Chunk Index", "Response"])

def append_row(csv_path, model_name, chunk_idx, response):
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([model_name, chunk_idx, response])

def extract_assistant_response(text: str) -> str:
    match = re.search(r"Assistant:\s*(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else ""

base_prompt = '''
You are a Video Understanding System.

You will be given :- 
- A short video clip of max 6 seconds
- A chunk index which will identify the position of this clip with respect to the full timeline

Rules :- 
- Do not assume anything beyond what is visible in the video
- Do hallucinate any future or past events 
- If the clip is unclear or ambigiuous, state the uncertainity 
- Focus on actions, objects, people, interactions, motion and scene changes 
- Be concise and precise 
- Try not to hallucinate and have issues with the end tokens. Meaning, do not keep on repeating information more than once in your response.

Question :- Describe me what is happening the video given to you. 
'''

for index, i in tqdm(enumerate(os.listdir(video_folder_path))):
    video_path = os.path.join(video_folder_path, i)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": base_prompt + f"\nChunk index for this video is {index}"},            
                            ]
                        },
                    ]
    llm_response = model1(messages)
    llm_response = extract_assistant_response(llm_response)
    #print(llm_response)
    append_row(csv_path, model_name, index, llm_response)