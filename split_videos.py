import PIL
import os

# Just a helper to split multiple videos in a folder into their frames at 5 fps.
# I couldn't figure out how to use pipe (|) in Windows with ffmpeg :(

for vid_file in os.listdir("vids"):
    os.system(f"ffmpeg -i vids\\{vid_file} -r 5 video-derived\\{vid_file}_%04d.jpg -hide_banner")