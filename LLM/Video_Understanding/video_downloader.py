import yt_dlp
import sys

def my_progress_hook(d):
    """
    This function is called automatically by yt-dlp during the download.
    'd' is a dictionary containing status information.
    """
    if d['status'] == 'downloading':
        p = d.get('_percent_str', '0%').replace('%','')
        
        try:
            percent = float(p)
        except ValueError:
            percent = 0.0

        speed = d.get('_speed_str', 'N/A')
        eta = d.get('_eta_str', 'N/A')

        print(f"UI UPDATE -> Progress: {percent}% | Speed: {speed} | ETA: {eta}")

    elif d['status'] == 'finished':
        print("UI UPDATE -> Download complete, now merging...")

def video_download(video_url):
    ydl_opts = {
        "format" : "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl" : "input.mp4",
        'progress_hooks': [my_progress_hook],
        'quiet' : True,
        "overwrites" : True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Starting Download for : {video_url}")
            ydl.download([video_url])
            print("\nDownload complete !\nSaved as video `Input.mp4`")
    except Exception as e:
        print(f"Error occured : {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ No video URL provided")
        sys.exit(1)

    video_url = sys.argv[1]
    video_download(video_url)