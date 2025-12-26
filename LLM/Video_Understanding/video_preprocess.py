import subprocess 
import os 

input_video = "input.mp4"
output_path = "processed_video.mp4"
output_dir = "video_chunks"
chunk_seconds = 6

os.makedirs(output_dir, exist_ok=True)

cmd_1 = ["ffmpeg", "-i", input_video, "-vf", 
         "scale=640:-1", "-c:a", "copy", output_path] #Resize the image to 640px on one side to reduce the dimensions for model input
subprocess.run(cmd_1, check=True)

cmd_2 = ["ffmpeg", "-i", input_video,
         "-c", "copy", "-map", "0",
         "-segment_time", str(chunk_seconds),
         "-f", "segment", "-reset_timestamps", "1",
         f"{output_dir}/chunk_%03d.mp4"]
subprocess.run(cmd_2, check=True)