import os

os.system(r'ffmpeg -framerate 24 -i "result_frames\result_%04d.jpg" -i audio\resultfract5.mp3 -c:a mp3 -c:v libx264 -y result_frames\result.mp4')