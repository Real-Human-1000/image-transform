from PIL import Image
import os
import numpy as np
from sklearn.linear_model import ElasticNet
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)
import matplotlib.pyplot as plt


target_img = Image.open("single_images\\ba.jpg")
target_img_small = target_img.resize((48,36))
target_arr_color = np.array(target_img_small).astype(float)
# target_arr = 0.299 * target_arr_color[:, :, 0] + 0.587 * target_arr_color[:, :, 1] + 0.114 * target_arr_color[:, :, 2]
# 0.229, 0.587, and 0.114 are used to get a more accurate depiction of luminosity
# according to what the human eye is sensitive to
target_arr = target_arr_color.reshape((target_arr_color.size, 1))  # vectorize
# target_arr /= np.linalg.norm(target_arr)
# target_arr -= 0.5


all_material_imgs = os.listdir("steamboat_frames")
material_imgs = []
N = 300  # Number of frames with which to build

"""
   material images x coefficients = target
[ img1 img2 img3 ]   [ coef1 ]   [ target ]
[ img1 img2 img3 ] x [ coef2 ] = [ target ]
[ img1 img2 img3 ]   [ coef3 ]   [ target ]

find coef vector using np.linalg.solve (or np.linalg.lstsq)
"""

for i in range(N):
    img = Image.open("steamboat_frames\\" + all_material_imgs[int(i / N * len(all_material_imgs))])
    img_array_color = np.array(img.filter(SMOOTH_MORE).resize((48,36))).astype(float)
    #img_array = 0.299 * img_array_color[:,:,0] + 0.587 * img_array_color[:,:,1] + 0.114 * img_array_color[:,:,2]
    material_imgs.append((all_material_imgs[int(i / N * len(all_material_imgs))], img_array_color))


# Build a single matrix of all images
imgs_arr = material_imgs[0][1].reshape((target_arr.size, 1))
for a in range(1,N):
    img_vec = material_imgs[a][1].reshape((target_arr.size, 1))
    # img_vec /= np.linalg.norm(img_vec)
    # img_vec -= 0.5
    imgs_arr = np.hstack((imgs_arr, img_vec))


# Calculate coefficients
def fit_least_squares(in_data, out_data):
    in_transpose = in_data.T
    square_inverse = np.linalg.pinv(in_transpose @ in_data)  # inv fails because singular matrix
    weights = square_inverse.T @ in_transpose @ out_data
    return weights  # maybe i'll just use lstsq actually


def soft_threshold(val, threshold):
    if val > threshold:
        return val - threshold
    if abs(val) < threshold:
        return 0
    if val < -threshold:
        return val + threshold


def fit_lasso(param, iters, in_data, out_data):
    # an implementation of the lasso algorithm, stolen from my old comp sci class
    weights = fit_least_squares(in_data, out_data)
    square_input = in_data.T @ in_data
    square_input[square_input == 0] = 1.0/(255.0*N)
    for iteration in range(iters):
        old_weights = weights.copy()
        for row in range(in_data.shape[1]):

            a_num_1 = (in_data.T @ out_data)[row,0]
            a_num_2 = (square_input[row,:] @ weights)[0]

            a_numerator = a_num_1 - a_num_2
            assert square_input[row, row] != 0
            a_for_row = a_numerator / (square_input[row, row])
            b_for_row = param / (2 * square_input[row,row])
            weights[row,0] = soft_threshold(weights[row,0] + a_for_row, b_for_row)

        print(iteration, np.sum(abs(weights - old_weights)))
        if np.sum(abs(weights - old_weights)) < 10**-4:  # stop if weights aren't changing much
            break
    return weights


# x = fit_lasso(100000, 1500, imgs_arr, target_arr)
# coefs = fit_least_squares(imgs_arr, target_arr)
model_alpha = 0.001
l1_ratio = 0.75  # 1.0 is LASSO, 0.1 is close to ridge
model = ElasticNet(alpha=model_alpha, l1_ratio=l1_ratio, selection='random', max_iter=10000, fit_intercept=True)
model.fit(imgs_arr, target_arr)
x = model.coef_
plt.plot(model.coef_, 'r')
plt.show()
print(x)
# print(coefs.size)
x[abs(x) < 5e-5] = 0
print(np.argwhere(x).shape[0])  # print number of non-zero elements

# Build nice image
coefs_sorted = list(zip(range(len(list(x))), list(x)))
coefs_sorted.sort(reverse=True, key=lambda tp: abs(tp[1]))
comp_arr = np.zeros(np.array(target_img).shape)
for coef_idx, const in coefs_sorted[:25]:
    this_img = np.array(Image.open("steamboat_frames\\" + material_imgs[coef_idx][0]).filter(SMOOTH_MORE))
    comp_arr += const * this_img

comp_arr = comp_arr + (np.mean(target_arr) - np.mean(comp_arr))
comp_arr = np.clip(comp_arr, 0, 255)
final_img = Image.fromarray(np.uint8(comp_arr))
final_img.save(f"result_frames\\model__material_frames!{N}__alpha!{model_alpha}__l1_ratio!{l1_ratio}__used_frames!{np.argwhere(x).shape[0]}.jpg")

# Build layers animation
# smallest = np.max(abs(x))
# for layer in range(np.argwhere(x).shape[0]):
#     coefs = np.copy(x)
#     coefs[abs(coefs) < smallest] = 0
#     print(f"Layer {layer}: {np.argwhere(coefs).shape[0]}")
#
#     target_img = Image.open("single_images\\ba.jpg")
#     comp_arr = np.zeros(np.array(target_img).shape)
#     for const_idx in range(len(list(coefs))):
#         this_img = np.array(Image.open("steamboat_frames\\" + material_imgs[const_idx][0]).filter(SMOOTH_MORE))
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