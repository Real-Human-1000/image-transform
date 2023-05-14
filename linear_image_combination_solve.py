from PIL import Image
import os
import numpy as np

target_img = Image.open("single_images\\ba.jpg").resize((48,36))
target_arr_color = np.array(target_img)
# target_arr = 0.299 * target_arr_color[:, :, 0] + 0.587 * target_arr_color[:, :, 1] + 0.114 * target_arr_color[:, :, 2]
# 0.229, 0.587, and 0.114 are used to get a more accurate depiction of luminosity
# according to what the human eye is sensitive to
target_arr = target_arr_color.reshape((target_arr_color.size, 1))  # vectorize
print(target_arr.shape)

all_material_imgs = os.listdir("other_frames")
material_imgs = []
N = 200  # Number of frames with which to build

"""
   material images x coefficients = target
[ img1 img2 img3 ]   [ coef1 ]   [ target ]
[ img1 img2 img3 ] x [ coef2 ] = [ target ]
[ img1 img2 img3 ]   [ coef3 ]   [ target ]

find coef vector using np.linalg.solve (or np.linalg.lstsq)
"""

for i in range(N):
    img = Image.open("other_frames\\" + all_material_imgs[int(i / N * len(all_material_imgs))]).resize((48,36))
    img_array_color = np.array(img)
    #img_array = 0.299 * img_array_color[:,:,0] + 0.587 * img_array_color[:,:,1] + 0.114 * img_array_color[:,:,2]
    material_imgs.append((all_material_imgs[int(i / N * len(all_material_imgs))], img_array_color))


# Build a single matrix of all images
imgs_arr = material_imgs[0][1].reshape((target_arr_color.size, 1))
for a in range(1,N):
    imgs_arr = np.hstack((imgs_arr, material_imgs[a][1].reshape((target_arr_color.size, 1))))


# Solve for the coefficients
# print(imgs_arr.shape)
# print(target_arr.shape)
x, residuals, rank, s = np.linalg.lstsq(imgs_arr, target_arr)
print(x)


# Build nice image
target_img = Image.open("single_images\\ba.jpg")
comp_arr = np.zeros(np.array(target_img).shape)
for const_idx in range(len(list(x))):
    this_img = np.array(Image.open("other_frames\\" + material_imgs[const_idx][0]))
    const = list(x)[const_idx]
    comp_arr += const * this_img

comp_arr = np.clip(comp_arr, 0, 255)
final_img = Image.fromarray(np.uint8(comp_arr))
final_img.save("result_frames\\solve.jpg")