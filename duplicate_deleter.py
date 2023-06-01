import imagehash
import os
from PIL import Image

# A handful of helpful functions for normalizing data and removing duplicate images


def split_gifs(folder):
    for filename in os.listdir(folder):
        name = ''.join(filename.split('.')[:-1])
        extension = filename.split('.')[-1].lower()
        if extension == "gif":
            im = Image.open(folder + "\\" + filename)
            for i in range(im.n_frames):
                im.seek(i)
                im.save(f"{folder}\\{name}_{i}.png")


def remove_duplicates(folder):
    p_hashes = []
    c_hashes = []
    for filename in os.listdir(folder):
        extension = filename.split('.')[-1].lower()

        if extension not in ("jpg", "png"):
            os.rename(folder + "\\" + filename, "gifs\\" + filename)
            continue

        this_img = Image.open(folder + "\\" + filename)
        p_hash = imagehash.phash(this_img)
        c_hash = imagehash.colorhash(this_img)

        p_duplicate = False
        c_duplicate = False
        for hash in p_hashes:
            if p_hash - hash < 3:
                p_duplicate = True
        for hash in c_hashes:
            if c_hash - hash < 3:
                c_duplicate = True

        if p_duplicate and c_duplicate:
            os.rename(folder + "\\" + filename, "duplicates\\" + filename)
        else:
            p_hashes.append(p_hash)
            c_hashes.append(c_hash)
    return len(p_hashes), len(c_hashes)


def square_images(folder):
    for filename in os.listdir(folder):
        im = Image.open(folder + "\\" + filename).convert("RGBA")
        bbox = im.getbbox()
        im_cropped = im.crop(bbox)
        im_copy = im.copy()
        # im_copy.resize((1,1))
        ave_color = im_copy.getpixel((1, 1))
        # ave_color = (255*round(ave_color[0]/255), 255*round(ave_color[1]/255), 255*round(ave_color[2]/255), 255*round(ave_color[3]/255))
        im_resized = Image.new("RGBA", (max(bbox[2], bbox[3]), max(bbox[2], bbox[3])), ave_color)
        im_resized.paste(im_cropped, ((im_resized.size[0] - im_cropped.size[0]) // 2, (im_resized.size[1] - im_cropped.size[1]) // 2))
        im_resized = im_resized.resize((512,512))
        im_resized.save("square\\" + ''.join(filename.split('.')[:-1]) + "_square.png")
        # os.remove(folder + "\\" + filename)


def img_size_histogram(folder):
    ratios = []
    for filename in os.listdir(folder):
        img = Image.open(folder + "\\" + filename)
        bbox = img.getbbox()
        dims = (bbox[2], bbox[3])
        ratios.append((max(dims) / min(dims), filename))
    histogram = {}
    for ratio in ratios:
        if ratio[0] not in histogram.keys():
            histogram[ratio[0]] = []
        histogram[ratio[0]].append(ratio[1])
    print(histogram)
    for key, val in histogram.items():
        if key > 1.78:
            for filename in val:
                os.rename(folder + "\\" + filename, "duplicates\\" + filename)

# remove_duplicates("video-derived")
# split_gifs("duplicates")
# remove_duplicates("duplicates")
square_images("video-derived")
# img_size_histogram("joe_images")
