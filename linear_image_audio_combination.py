# Combining lessons from the Delay Lama Transform and the other approaches in this project
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFilter import (
    BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
    EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)
import os
import shutil
import numpy as np
import scipy
from scipy.optimize import nnls
from sklearn.linear_model import ElasticNet
import scipy.io.wavfile as wavfile
from scipy.interpolate import make_interp_spline, CubicSpline
import wave
import matplotlib.pyplot as plt

# Strategy: use linear model (solve, lasso, elastic net, etc.) but add audio spectrum info to the ends of the frame vectors
# Duration of an audio slice has to be the same as the duration of a frame
# This is going to be quite inconvenient because now we can't delete duplicate frames because they might have different audio
# Also we're going to have audio problems because Steamboat only goes up to like 6000 and Bad Apple goes to like 15000
# We can low-pass Bad Apple to be mostly below 6000 and it doesn't sound too bad
# The best organizational strategy might be to allow this program to handle the splitting of the video so that it can control fps

# it's really essential that everything is done at 5 fps (or at least the same fps)
# Now we can separate image fps from audio processing fps (true fps)
fps = 24
img_fps_ratio = 0.5  # img fps will be fps * img_fps_ratio (if it's 0.5, then half of frames will be uniquely rendered)
working_size = (48,36)  # it's cheaper to work with images at a lower resolution
wavsamplerate = 44100

target_frame_dir = "badapple_frames"
target_audio_file = "audio\\badapplevoiceisolate6000(3000)mono250.wav"  # currently best results from mono250
material_frame_dir = "steamboat_frames"
material_audio_file = "audio\\steamboatmono.wav"

textfont = ImageFont.truetype("fonts\\Hack-Regular.ttf", 20)


def get_working_image_vector(filename):
    # Opens, resizes, filters, and reshapes an image so that it can be used
    img = Image.open(filename).resize(working_size)
    img_array_color = np.array(img.filter(SMOOTH_MORE)).astype(float)
    img_array = 0.299 * img_array_color[:, :, 0] + 0.587 * img_array_color[:, :, 1] + 0.114 * img_array_color[:, :, 2]
    img_vector = img_array.reshape(working_size[0] * working_size[1])
    return img_vector


def spectrify(audio_file):
    # given a filename, return f, t, Sxx

    # Load audio data into a numpy vector
    # Read file to get buffer
    audio_wave = wave.open(audio_file)
    samples = audio_wave.getnframes()
    audio_frames = audio_wave.readframes(samples)
    num_frames = audio_wave.getnframes()
    rate = audio_wave.getframerate()
    duration = num_frames / float(rate)

    # https://stackoverflow.com/questions/16778878/python-write-a-wav-file-into-numpy-float-array
    # Convert buffer to float32 using NumPy
    audio_as_np_int16 = np.frombuffer(audio_frames, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

    # Normalise float32 array so that values are between -1.0 and +1.0
    max_int16 = 2**15
    audio_vec = audio_as_np_float32 / max_int16

    win = scipy.signal.get_window('hann', 2048)
    f, t, Sxx = scipy.signal.spectrogram(audio_vec, wavsamplerate, mode='psd', window=win)

    return f, t, Sxx, duration, audio_vec


def create_edited_frame(main_image, video_images, audio_image):
    # BadApple is usually 4:3, but we want to convert it to 16:9 w/ the same height by padding the right side with info
    # Inputs:
    #  - main_image: PIL image, 4:3 aspect ratio
    #  - video_images: list of 8 PIL images, 4:3 aspect ratio each
    #  - audio_image: PIL image, 4:3 aspect ratio
    # Outputs:
    #  - PIL image, 16:9 aspect ratio
    img_width = int(16/9 * main_image.size[1])
    canvas = Image.new(mode="RGB", size=(img_width, main_image.size[1]))
    # Paste in main image
    canvas.paste(main_image, box=(0,0))
    # Paste in video_images
    small_width = (img_width - main_image.size[0]) / 2
    small_height = 3 / 4 * small_width
    # There's something weird going on with paste(), so we will resize all of the images first
    for row in range(4):
        for col in range(2):
            if video_images[row*2+col] is not None:
                canvas.paste(video_images[row*2+col].resize((int(small_width), int(small_height))), box=(main_image.size[0] + int(small_width) * col, int(small_height * (row + 0.7))))
    # Paste in audio_image
    canvas.paste(audio_image.resize((int(small_width), int(small_height))), box=(img_width - int(small_width), main_image.size[1] - int(small_height * 1.2)))
    # Draw text
    d = ImageDraw.Draw(canvas)
    d.text((main_image.size[0] + 10, 10), "Video from:", font=textfont, fill=(255, 255, 255, 255))
    d.text((main_image.size[0] + 10, main_image.size[1] - int(small_height * 1.1)), "Audio\nfrom:", font=textfont, fill=(255, 255, 255, 255))
    return canvas


def overwrite_audio_image(frame, new_audio_image):
    # Inputs:
    #  - frame: rendered frame, PIL Image, 16:9
    #  - new_audio_image: used to replace old audio_image in frame; 4:3
    # Relies on the fact that all images are the same size as new_audio_image before being rendered into frames
    small_width = (frame.size[0] - new_audio_image.size[0]) / 2
    small_height = 3 / 4 * small_width
    frame.paste(new_audio_image.resize((int(small_width), int(small_height))), box=(frame.size[0] - int(small_width), new_audio_image.size[1] - int(small_height * 1.2)))
    return frame


def logify(arr):
    # Log on array with a term to prevent log(0) = infinity
    return np.log(1e6 * arr + np.ones_like(arr) * np.e) - 1


# Perspective projection resources from:
# https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
# https://web.archive.org/web/20150222120106/xenia.media.mit.edu/~cwren/interpolator/

def find_coeffs(pa, pb):
    # Converts transformation between initial points pa and post-transformation points pb into coefficients for transformation
    # Inputs: pa (list of 4 2-tuples, after transformation), pb (list of 4 2-tuples, before transformation)
    # Returns: np array of 8 coefficients for transformation to by used by Image.transform()
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float32)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def project_points(p3d):
    # Returns 2D coords of 3D point projected onto the screen
    # Input: p3d (np array with 3 rows and n columns)
    # Output: np array with 2 rows and n columns
    return np.divide(p3d[0:2, :], np.vstack((p3d[2,:], p3d[2,:])))


def rot3D(yaw, pitch, roll):
    # Returns 3D rotation matrix
    # Inputs: yaw, pitch, roll (angles in radians)
    # Output: matrix to multiply a column vector with
    yawMat = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    pitchMat = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    rollMat = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    return yawMat @ pitchMat @ rollMat


def stack_images(top_img, other_imgs):
    # Creates a stack graphic of images
    # Inputs: top_img (PIL image), other_imgs (list of PIL images)
    # Output: PIL image

    # Top image will be on top, fully opaque
    # Other images will be underneath and half transparent
    final_img = None

    for i in range(len(other_imgs)+1):
        if i == len(other_imgs):
            img = top_img.convert('RGBA')
        else:
            img = other_imgs[i].convert(mode='RGBA')
            img.putalpha(127)
        width, height = img.size

        initial_3d_points = np.array([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0]]).T
        rotated_points = rot3D(0.3,0.2,-0.7) @ initial_3d_points  # imagine airplane in plane of screen in this direction: >--)->
        y = (1 + 0.35*len(other_imgs)) - i * 0.5  # works well with 15 images: (1 + 0.1*len(other_imgs)) - i * 0.2
        final_3d_points = rotated_points + np.array([[0,y,2], [0,y,2], [0,y,2], [0,y,2]]).T
        final_3d_points = rot3D(0,0,0.28) @ final_3d_points  # camera rotation
        proj_points = project_points(final_3d_points)
        rescaled_points = np.vstack(((proj_points[0,:] + 1) / 2 * width, (proj_points[1,:] + 1) / 2 * height))

        new_points = [(rescaled_points[0,col], rescaled_points[1,col]) for col in range(rescaled_points.shape[1])]

        coeffs = find_coeffs(new_points,[(0, 0), (width, 0), (width, height), (0, height)])

        transformed_img = img.transform(size=(width, height), method=Image.PERSPECTIVE, data=coeffs,
                                        resample=Image.BICUBIC, fillcolor=(0,0,0,0))

        if final_img is None:
            final_img = transformed_img
        else:
            final_img = Image.alpha_composite(final_img, transformed_img)

    return final_img


# Get spectrum of material audio
material_f, material_t, material_Sxx, material_duration, material_audio = spectrify(material_audio_file)

# Load all frames and spectra into two massive matrices
print("Loading material")
material_vector_len = working_size[0] * working_size[1]
num_audio_material_frames = int(material_duration * fps)
num_img_material_frames = int(material_duration * fps * img_fps_ratio)
material_arr = np.zeros((working_size[0] * working_size[1], num_img_material_frames))
material_spectra = np.zeros((len(material_f), num_audio_material_frames))

for mf_idx in range(1, int(material_duration * fps) - 2):  # some weird overlap issue --> -2 last frames

    spectrum_slice_start = (mf_idx * 1/fps) * len(material_t) / material_duration
    spectrum_slice_end = int(spectrum_slice_start + 1/fps * len(material_t) / material_duration)
    spectrum_slice_start = int(spectrum_slice_start)

    if spectrum_slice_start == spectrum_slice_end:
        spectrum_slice_end += 1
    assert spectrum_slice_start != spectrum_slice_end

    spectrum_vector = material_Sxx[:, spectrum_slice_start : spectrum_slice_end].mean(axis=1)
    material_spectra[:, mf_idx] = 1.0 * logify(spectrum_vector)

    if mf_idx % 1/img_fps_ratio == 0:
        img_vector = get_working_image_vector(material_frame_dir + "\\" + os.listdir(material_frame_dir)[int(mf_idx * img_fps_ratio)])
        material_arr[:, int(mf_idx * img_fps_ratio)] = 1.0 * img_vector


print(f"Material arr shape: {material_arr.shape}")
print(f"Material spectra shape: {material_spectra.shape}")


# Go through each frame of the target video and calculate the necessary sum of material frames
target_f, target_t, target_Sxx, target_duration, target_audio = spectrify(target_audio_file)

comp_audio = np.zeros(int(wavsamplerate * target_duration))
for tf_idx in range(0*fps, 60*fps):
    print("Processing " + str(tf_idx / fps) + " sec (" + str(100*(tf_idx - 0*fps) / (60*fps - 0*fps)) + "%)")

    target_spectrum_slice_start = (tf_idx * 1/fps) * len(target_t) / target_duration
    target_spectrum_slice_end = int(target_spectrum_slice_start + 1/fps * len(target_t) / target_duration)
    target_spectrum_slice_start = int(target_spectrum_slice_start)

    if target_spectrum_slice_start == target_spectrum_slice_end:
        target_spectrum_slice_end += 1
    assert target_spectrum_slice_start != target_spectrum_slice_end

    target_spectrum_vector = target_Sxx[:, target_spectrum_slice_start: target_spectrum_slice_end].mean(axis=1)
    target_spectrum_vector[abs(target_spectrum_vector) < 1e-10] = 0

    target_spectrum = 1 * logify(target_spectrum_vector)

    # Calculate audio separately
    comp_slice_start = round(tf_idx / fps * wavsamplerate)
    comp_slice_end = round((tf_idx + 1) / fps * wavsamplerate)

    # Calculate audio using non-negative least squares (since we can't subtract frequencies)
    # ElasticNet doesn't seem to be working super well but the acoefs aren't generally very large (~3 max)
    # Correction: if the audio is weird (like unprocessed badapple.wav), then the coefs can be gigantic (90k)
    # acoefs, norm = nnls(material_spectra, target_spectrum)  # positive only
    # acoefs_sorted = list(zip(range(len(acoefs)), acoefs))
    # acoefs_sorted.sort(reverse=True, key=lambda tp: abs(tp[1]) )
    # print(acoefs_sorted[:15])

    # ElasticNet: doesn't seem to work...?
    # audio_model_alpha = 1
    # audio_l1_ratio = 0.75  # 1.0 is LASSO, 0.1 is close to ridge
    # model = ElasticNet(alpha=audio_model_alpha, l1_ratio=audio_l1_ratio, selection='random', max_iter=10000, fit_intercept=False, positive=True)
    # model.fit(material_spectra, target_spectrum)
    # acoefs = model.coef_

    # Single-column scaling (one audio source per frame; works decently well for high framerates and is more interesting)
    # (Also is way faster than solving or regression)
    normalized_material = (material_spectra / (np.linalg.norm(material_spectra, axis=0) + 1e-10))
    normalized_target = target_spectrum / (np.linalg.norm(target_spectrum) + 1e-10)
    normalized_target = np.array([normalized_target]).T
    diffs = normalized_material - normalized_target
    audio_closest_idx = np.argmin(np.abs(np.linalg.norm(diffs, axis=0)))
    audio_image_idx = int(audio_closest_idx / material_spectra.shape[1] * material_arr.shape[1])  # making an assumption that the audio duration is the same as the video
    const = 0
    if np.linalg.norm(material_spectra[:,audio_closest_idx]) != 0:
        const = np.linalg.norm(target_spectrum) / np.linalg.norm(material_spectra[:,audio_closest_idx])
    # print(const)
    acoefs_sorted = [(audio_closest_idx, const)]
    scaled_closest = normalized_material[:,audio_closest_idx] * np.linalg.norm(target_spectrum)
    # if np.linalg.norm(target_spectrum) > 1e-10:
    #    plt.plot(material_spectra[:, closest_idx], label='unscaled closest')
    #    plt.plot(target_spectrum, label='target')
    #    plt.plot(scaled_closest, label='scaled closest')
    #    plt.legend()
    #    plt.show()

    # Build composite audio (works for any audio construction method)
    for acoef_idx, const in acoefs_sorted:
        # Using round() because multiplying/dividing large numbers will sometimes output something like 100.9999999998
        material_slice_start = round(acoef_idx / fps * wavsamplerate)
        material_slice_end = round((acoef_idx+1) / fps * wavsamplerate)
        # print("Slices: ", comp_slice_start, comp_slice_end, material_slice_start, material_slice_end)
        # I think we can afford this crazy sloppiness because of the audio smoothing that happens next
        material_slice_end += (int(comp_slice_end) - int(comp_slice_start)) - (material_slice_end - material_slice_start)
        comp_audio[int(comp_slice_start) : int(comp_slice_end)] += const * material_audio[int(material_slice_start) : int(material_slice_end)]

    # Interpolate audio to remove clicks
    # Overwrite the last few samples of the previous audio slice and the first few samples of this audio slice
    # Use a B-spline whose 3 points are -5, -4, -3 of the previous slice and 2, 3, 4 of the current slice (may be inaccurate)
    if comp_slice_start > 17 and not comp_slice_end > len(comp_audio)-15 and np.sum(np.abs(comp_audio[comp_slice_start-5:comp_slice_end+6])) > 0:
        deriv_l = (comp_audio[comp_slice_start-14] - comp_audio[comp_slice_start-15]) / 2
        # Experimental multi-point regression technique -- doesn't work significantly better than splines
        # line_l, res, rank, s = np.linalg.lstsq(np.array([[0,1],[1,1],[2,1],[3,1]]), comp_audio[comp_slice_start-17:comp_slice_start-13].T, rcond=None)
        # deriv_l = line_l[0]
        deriv_r = (comp_audio[comp_slice_start+14] - comp_audio[comp_slice_start+13]) / 2
        bspline = make_interp_spline((-14, -13, -12, -11, -10, -9, -8, -7, -6, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13),
                                     np.hstack((comp_audio[comp_slice_start-14:comp_slice_start-5],
                                                np.array([(comp_audio[comp_slice_start-6] + comp_audio[comp_slice_start+5]) / 2]),  # will hopefully discourage generation of new peaks
                                                comp_audio[comp_slice_start+5:comp_slice_start+14])),
                                     bc_type=([(1, deriv_l)], [(1, deriv_r)]))
        for off in range(-5,5):
            comp_audio[int(comp_slice_start+off)] = bspline(off)

    if comp_slice_start > comp_slice_end - comp_slice_start:
        # Idea: to reduce sudden spikes in loudness, normalize this audio section to at most 2x the RMS of the last section
        # If the previous section was all zero, though, I guess this one can have any RMS
        last_rms = np.linalg.norm(comp_audio[int(comp_slice_start - (comp_slice_end - comp_slice_start)):int(comp_slice_start)])
        this_rms = np.linalg.norm(comp_audio[int(comp_slice_start):int(comp_slice_end)])
        print(last_rms, this_rms)
        if this_rms > 5 / (last_rms + 1):
            comp_audio[int(comp_slice_start):int(comp_slice_end)] *= (5 / (last_rms + 1))/this_rms


    # Calculate frame if the img_fps matches up; otherwise, copy the last-generated frame
    # if tf_idx % (1/img_fps_ratio) == 0:
    #     print("Image frame")
    #     # Load target image vector
    #     target_vector = get_working_image_vector(target_frame_dir + "\\" + os.listdir(target_frame_dir)[int(tf_idx * img_fps_ratio)])
    #
    #     if np.sum(target_vector) < 1e-5:
    #         stack_list = [None]*8
    #         comp_arr = np.zeros((360, 480, 3))
    #     else:
    #
    #         # ------Solve for coefficients-------
    #         # Solve directly with lstsq
    #         # x, residuals, rank, s = np.linalg.lstsq(material_arr, target_vector, rcond=None)
    #         # x, rnorm = nnls(imgs_arr, target_arr.reshape(target_arr.size,))  # positive only
    #
    #         # Solve using ElasticNet to get more reasonable coefficients
    #         model_alpha = 15  # 10 for images
    #         l1_ratio = 0.8  # 1.0 is LASSO, 0.1 is close to ridge
    #         model = ElasticNet(alpha=model_alpha, l1_ratio=l1_ratio, fit_intercept=False, selection='random', max_iter=10000)
    #         model.fit(material_arr, target_vector)
    #         x = model.coef_
    #
    #         x[abs(x) < 1e-10] = 0
    #         # print(np.argwhere(x).shape[0])
    #
    #         # Build nice composite image
    #         target_img = Image.open(target_frame_dir + "\\" + os.listdir(target_frame_dir)[int(tf_idx * img_fps_ratio)])
    #         comp_arr = np.zeros((360, 480, 3))
    #         coefs_sorted = list(zip(range(len(list(x))), list(x)))
    #         coefs_sorted.sort(reverse=True, key=lambda tp: abs(tp[1]))
    #         stack_list = []  # list for stacked images
    #         print(coefs_sorted[:10])
    #         for coef_idx, const in coefs_sorted[:65]:
    #             if const == 0:
    #                 continue
    #             this_img = np.array(Image.open(material_frame_dir + "\\" + os.listdir(material_frame_dir)[coef_idx]).filter(SMOOTH_MORE), dtype=np.float32)
    #             # this_img *= const
    #             comp_arr += this_img * const
    #             if const < 0:
    #                 this_img = 255 - this_img
    #             # Force the const to be either 0.5, 1.0, or 1.5 for visual appeal
    #             # if np.abs(const) < 0.2:
    #             #     const = 0.5
    #             # elif np.abs(const) > 0.3:
    #             #     const = 1.5
    #             # else:
    #             #     const = 1.0
    #             # this_img *= np.abs(const)
    #             # if np.mean(this_img) > 255:
    #             #     this_img = this_img - np.mean(this_img) + 255
    #             #     # img [260, 270, 280] mean is 270 --> img-mean = [-10, 0, 10] --> img-mean+255 = [245, 255, 265]
    #             # if np.mean(this_img) < 0:
    #             #     this_img -= np.mean(this_img)
    #             #     # img [-10, -20, -30] mean is -20 --> img-mean = [10, 0, -10]
    #             # Need to ensure that at least the average pixel is visible (value clamped between 0 and 255)
    #             stack_list.append(Image.fromarray(np.uint8(np.clip(this_img, 0, 255))))
    #
    #     # Create and save final frame
    #     comp_arr = comp_arr + (np.mean(target_vector) - np.mean(comp_arr))
    #     comp_arr = np.clip(comp_arr, 0, 255)
    #     final_img = Image.fromarray(np.uint8(comp_arr)).convert(mode="RGB")
    #     # video_images = [Image.open(material_frame_dir + "\\" + os.listdir(material_frame_dir)[coef_idx]).convert(mode="RGB").filter(SMOOTH_MORE) for coef_idx, const in coefs_sorted[:8]]
    #     audio_image = Image.open(material_frame_dir + "\\" + os.listdir(material_frame_dir)[audio_image_idx]).convert(mode="RGB").filter(SMOOTH_MORE)
    #     final_frame = create_edited_frame(final_img, [stack_list[i] if i < len(stack_list) else None for i in range(8)], audio_image)
    #
    #     # Create stacked final frame
    #     # final_frame = stack_images(final_img, stack_list[:10]).convert('RGB')  # the first 5-10 seem to be boring
    #
    #     final_frame.save(f"result_frames\\result_{(len(os.listdir('result_frames')) + 1):04d}.jpg")
    #
    # else:
    #     # Just load the most recent frame and overwrite its audio image
    #     print("Audio-only frame")
    #     previous_frame = Image.open(f"result_frames\\result_{(len(os.listdir('result_frames')) + 0):04d}.jpg")
    #     audio_image = Image.open(material_frame_dir + "\\" + os.listdir(material_frame_dir)[audio_image_idx]).convert(mode="RGB").filter(SMOOTH_MORE)
    #     new_frame = overwrite_audio_image(previous_frame, audio_image)
    #     # new_frame = previous_frame
    #     new_frame.save(f"result_frames\\result_{(len(os.listdir('result_frames')) + 1):04d}.jpg")

# Save audio into wav file
# https://stackoverflow.com/questions/10357992/how-to-generate-audio-from-a-numpy-array
scaled = np.int16(comp_audio / np.max(np.abs(comp_audio)) * 32767)
wavfile.write('audio\\RESULT.wav', wavsamplerate, scaled.astype(np.int16))

# Oh noes
# I just realized that the audio can't be subtracted and the images can't be solved with only positive coefficients
# So we can't solve for both at the same time
# Well that sucks
# Although it does save me a lot of trouble
# FAILED: Embrace failure
# We will persevere!