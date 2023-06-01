from PIL import Image
import os
import numpy as np


target_img = Image.open("single_images\\ba.jpg").resize((48,36))
target_arr_color = np.array(target_img).astype(float)
# target_arr = 0.299 * target_arr_color[:, :, 0] + 0.587 * target_arr_color[:, :, 1] + 0.114 * target_arr_color[:, :, 2]
# 0.229, 0.587, and 0.114 are used to get a more accurate depiction of luminosity
# according to what the human eye is sensitive to
target_arr = target_arr_color.reshape((target_arr_color.size, 1))  # vectorize

all_material_imgs = os.listdir("other_frames")
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
    img = Image.open("other_frames\\" + all_material_imgs[int(i / N * len(all_material_imgs))]).resize((48,36))
    img_array_color = np.array(img).astype(float)
    #img_array = 0.299 * img_array_color[:,:,0] + 0.587 * img_array_color[:,:,1] + 0.114 * img_array_color[:,:,2]
    material_imgs.append((all_material_imgs[int(i / N * len(all_material_imgs))], img_array_color))


# Build a single matrix of all images
imgs_arr = material_imgs[0][1].reshape((target_arr.size, 1))
for a in range(1,N):
    imgs_arr = np.hstack((imgs_arr, material_imgs[a][1].reshape((target_arr.size, 1))))


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
        if np.sum(abs(weights - old_weights)) < 10**-4:
            break
    return weights


x = fit_lasso(100000, 1500, imgs_arr, target_arr)
# coefs = fit_least_squares(imgs_arr, target_arr)
print(x)
# print(coefs.size)
x[abs(x) < 5e-2] = 0
print(np.argwhere(x).shape[0])

# Build nice image
# target_img = Image.open("single_images\\kirby.jpg")
# comp_arr = np.zeros(np.array(target_img).shape)
# for coef_idx in range(len(list(coefs))):
#     this_img = np.array(Image.open("badapple_frames\\" + material_imgs[coef_idx][0]))
#     const = list(coefs)[coef_idx]
#     comp_arr += const * this_img
#
# comp_arr = np.clip(comp_arr, 0, 255)
# final_img = Image.fromarray(np.uint8(comp_arr))
# final_img.save(f"result_frames\\lasso_{N}.jpg")

# Build layers animation
smallest = np.max(abs(x))
for layer in range(np.argwhere(x).shape[0]):
    coefs = np.copy(x)
    coefs[abs(coefs) < smallest] = 0
    print(f"Layer {layer}: {np.argwhere(coefs).shape[0]}")

    target_img = Image.open("single_images\\ba.jpg")
    comp_arr = np.zeros(np.array(target_img).shape)
    for const_idx in range(len(list(coefs))):
        this_img = np.array(Image.open("other_frames\\" + material_imgs[const_idx][0]))
        const = list(coefs)[const_idx]
        comp_arr += const * this_img

    comp_arr = comp_arr + (np.mean(target_arr) - np.mean(comp_arr))
    comp_arr = np.clip(comp_arr, 0, 255)
    final_img = Image.fromarray(np.uint8(comp_arr))
    for copies in range(int(1+19/((layer+1)**0.75))):
        final_img.save(f"result_frames\\result_{(len(os.listdir('result_frames'))+1):04d}.jpg")

    temp = np.copy(x)
    temp[abs(temp) >= abs(smallest)] = 0
    smallest = np.max(abs(temp))