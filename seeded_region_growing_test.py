# %%  #imports

import numpy as np
import skimage.io as sk
from Functions import image_processing as ip
from Functions import seed_detection as sd



if __name__ == '__main__':
    image = sk.imread("Data/N2DH-GOWT1/img/t01.tif")  # Bild laden

    img_small = image[300:350, 400:450]
    ip.show_image(img_small, 15, 8)
    img_r = sd.seeds(img_small, 0.1, 1)
    ip.show_image(img_r, 15, 8)
    img_r = sd.seed_merging(img_r)
    ip.show_image(img_r, 15, 8)

    img_r = region_growing(img_small, img_r)
    ip.show_image(img_r, 15, 8)
