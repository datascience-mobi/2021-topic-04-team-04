
#  from Functions import image_processing as ip
import skimage.io as sk
#  import numpy as np
from Functions import unseeded_region_growing as urg
from PIL import Image

image = sk.imread("Data/N2DH-GOWT1/img/t01.tif")
#  img_r = ip.img_resize(image, 500, 500)
#  ip.show_image(img_r, 15, 8)

#  img_small = image[0:500, 0:500]
#  ip.show_image(img_small, 15, 8)


img_result = urg.unseeded_region_growing_algorithm(image, (0, 0), 5)

im = Image.fromarray(img_result)
im.save("Result_Pictures/Unseeded_Region_Growing/N2DH-GOWT1/urg_t01.tif")
