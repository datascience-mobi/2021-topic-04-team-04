import skimage.io as sk
import numpy as np
from Functions import image_processing as ip
from Functions import seed_detection as sd
from PIL import Image
from Functions import seeded_region_growing as srg


image_intensity = sk.imread("../Data/N2DH-GOWT1/img/t02.tif")  # load image
#image_intensity = image_intensity[300:350, 450:500]
image_r = sd.seeds(image_intensity, 0.1, 1)
image_r = sd.seed_merging(image_r)
image_seeds = Image.fromarray(image_r)
image_seeds.save("../Result_Pictures/Seeded_Region_Growing/N2DH-GOWT1/srg_t02_seeds.tif")

image_r = srg.region_growing(image_intensity, image_r)
ip.show_image(image_r, 15, 8)

im = Image.fromarray(image_r)
im.save("../Result_Pictures/Seeded_Region_Growing/N2DH-GOWT1/srg_t02.tif")