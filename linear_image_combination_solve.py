from PIL import Image
import os
import numpy as np
from scipy.optimize import nnls
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)


sigmoid = lambda x : 255 / (1 + np.exp(-20 * (x/255 - 0.3)))


target_dir = "badapple_frames"
material_dir = "steamboat_frames"

all_material_imgs = os.listdir(material_dir)
material_imgs = []
N = 800  # Number of frames with which to build
working_size = (480//4,360//4)

"""
   material images x coefficients = target
[ img1 img2 img3 ]   [ coef1 ]   [ target ]
[ img1 img2 img3 ] x [ coef2 ] = [ target ]
[ img1 img2 img3 ]   [ coef3 ]   [ target ]

find coef vector using np.linalg.solve (or np.linalg.lstsq)
"""

for i in range(N):
    img = Image.open(material_dir + "\\" + all_material_imgs[int(i / N * len(all_material_imgs))]).resize(working_size)
    img_array_color = np.array(img.filter(SMOOTH_MORE)).astype(float)
    #img_array = 0.299 * img_array_color[:,:,0] + 0.587 * img_array_color[:,:,1] + 0.114 * img_array_color[:,:,2]
    material_imgs.append((all_material_imgs[int(i / N * len(all_material_imgs))], img_array_color))


# Build a single matrix of all images
imgs_arr = material_imgs[0][1].reshape((material_imgs[0][1].size, 1))
for a in range(1,N):
    imgs_arr = np.hstack((imgs_arr, material_imgs[a][1].reshape((material_imgs[0][1].size, 1))))


for target_name in os.listdir(target_dir):
    print(target_name)
    # Get target image
    target_img = Image.open(target_dir + "\\" + target_name).resize(working_size)
    target_arr_color = np.array(target_img).astype(float)
    # target_arr = 0.299 * target_arr_color[:, :, 0] + 0.587 * target_arr_color[:, :, 1] + 0.114 * target_arr_color[:, :, 2]
    # 0.229, 0.587, and 0.114 are used to get a more accurate depiction of luminosity
    # according to what the human eye is sensitive to
    target_arr = target_arr_color.reshape((target_arr_color.size, 1))  # vectorize


    # Solve for the coefficients
    x, residuals, rank, s = np.linalg.lstsq(imgs_arr, target_arr)
    # x, rnorm = nnls(imgs_arr, target_arr.reshape(target_arr.size,))  # positive only
    x[abs(x) < 2e-2] = 0
    print(np.argwhere(x).shape[0])

    # Build nice image
    target_img = Image.open(target_dir + "\\" + target_name)
    comp_arr = np.zeros(np.array(target_img).shape)
    for coef_idx in range(len(list(x))):
        this_img = np.array(Image.open(material_dir + "\\" + material_imgs[coef_idx][0]).filter(SMOOTH_MORE))
        const = list(x)[coef_idx]
        comp_arr += const * this_img

    # print(np.mean(target_arr), np.mean(comp_arr))
    comp_arr = comp_arr + (np.mean(target_arr) - np.mean(comp_arr))
    comp_arr = np.clip(comp_arr, 0, 255)
    final_img = Image.fromarray(np.uint8(comp_arr))
    final_img.save(f"result_frames\\result_{(len(os.listdir('result_frames'))+1):04d}.jpg")


# Build layer animation
# smallest = np.max(abs(x))
# for layer in range(np.argwhere(x).shape[0]):
#     coefs = np.copy(x)
#     coefs[abs(coefs) < smallest] = 0
#     print(f"Layer {layer}: {np.argwhere(coefs).shape[0]}")
#
#     target_img = Image.open("single_images\\ba.jpg")
#     comp_arr = np.zeros(np.array(target_img).shape)
#     for const_idx in range(len(list(coefs))):
#         this_img = np.array(Image.open("other_frames\\" + material_imgs[const_idx][0]))
#         const = list(coefs)[const_idx]
#         comp_arr += const * this_img
#
#     comp_arr = comp_arr + (np.mean(target_arr) - np.mean(comp_arr))
#     comp_arr = np.clip(comp_arr, 0, 255)
#     final_img = Image.fromarray(np.uint8(comp_arr))
#     for copies in range(int(1+19/((layer+1)**0.75))):
#         final_img.save(f"result_frames\\result_{(len(os.listdir('result_frames'))+1):04d}.jpg")
#
#     temp = np.copy(x)
#     temp[abs(temp) >= abs(smallest)] = 0
#     smallest = np.max(abs(temp))