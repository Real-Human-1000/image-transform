#Import required image modules
from PIL import Image, ImageFilter
from math import exp
#Import all the enhancement filter from pillow
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)


smoothstep = lambda x : (-20 * (x/255)**7 + 70 * (x/255)**6 - 84 * (x/255)**5 + 35 * (x/255)**4) * 255
sigmoid = lambda x : 255 / (1 + exp(-20 * (x/255 - 0.3)))


#Create image object
img = Image.open('steamboat_frames\\steamboat360resize.mp4_1288.jpg').resize((48,36))
# img.show()
#Applying the filter
img1 = img.filter(SMOOTH_MORE)

# img1 = img.convert('L').point(sigmoid)
img1 = img1.filter(ImageFilter.BoxBlur(radius=1))
img1.show()
# img1.save("result_frames\\before.png")

