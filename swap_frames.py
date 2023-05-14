import imagehash
from PIL import Image, ImageFilter
import os
import shutil

# ffmpeg -r 10 -i result_%04.jpg -vcodec mpeg4 result.mp4

badapple_frames = [(name, imagehash.phash(Image.open("badapple_frames\\" + name).filter(ImageFilter.FIND_EDGES))) for name in os.listdir("badapple_frames")]
print("Bad Apple!! Loaded")
other_frames = [(name, imagehash.phash(Image.open("other_frames\\" + name).filter(ImageFilter.FIND_EDGES))) for name in os.listdir("other_frames")]
print("Other Media Loaded")

swap = []

for ba_idx in range(len(badapple_frames)):
    distances = [badapple_frames[ba_idx][1] - o_frame[1] for o_frame in other_frames]
    print(min(distances))
    swap.append(distances.index(min(distances)))

for s_idx in range(len(swap)):
    ba_frame = badapple_frames[s_idx]
    number_jpg = ba_frame[0].split("_")[1]
    print(number_jpg)
    other_frame = other_frames[swap[s_idx]]
    shutil.copyfile("other_frames\\" + other_frame[0], "result_frames\\result_" + number_jpg)
