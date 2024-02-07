import PIL
import os

# Just a helper to split multiple videos in a folder into their frames at 5 fps.
# I couldn't figure out how to use pipe (|) in Windows with ffmpeg :(

# https://video.stackexchange.com/questions/4563/how-can-i-crop-a-video-with-ffmpeg
# ffmpeg -i in.mp4 -filter:v "crop=out_w:out_h:x:y" out.mp4

# https://superuser.com/questions/624563/how-to-resize-a-video-to-make-it-smaller-with-ffmpeg
# ffmpeg -i input.avi -s 720x480 -c:a copy output.mkv

for vid_file in os.listdir("steamboat_frames"):
    os.system(f"ffmpeg -i steamboat_frames\\{vid_file} -qscale:v 2 -r 5 steamboat_frames\\{vid_file}_%04d.jpg -hide_banner")