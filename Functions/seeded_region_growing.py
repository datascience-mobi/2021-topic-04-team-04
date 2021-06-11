import skimage.io as sk
import numpy as np
from Functions import image_processing as ip
from Functions import seed_detection as sd


# import math as m
# import matplotlib.pyplot as plt

def find_neighbors(reg):
    neighbors = []
    for p in np.ndindex(reg.shape):
        if reg[p] != 0:  # Pixels with region
            if p[0] > 0:  # Add neighbours to list T, up
                a = (p[0] - 1, p[1])
                if reg[a] == 0 and a not in neighbors:
                    neighbors.append(a)
            if p[0] < reg.shape[0] - 1:  # Add neighbours to list T, down
                b = (p[0] + 1, p[1])
                if reg[b] == 0 and b not in neighbors:
                    neighbors.append(b)
            if p[1] > 0:  # Add neighbours to list T, left
                c = (p[0], p[1] - 1)
                if reg[c] == 0 and c not in neighbors:
                    neighbors.append(c)
            if p[1] < reg.shape[1] - 1:  # Add neighbours to list T, right
                d = (p[0], p[1] + 1)
                if reg[d] == 0 and d not in neighbors:
                    neighbors.append(d)
    return neighbors


def add_neighbors(img, p):  # p describes pixel for which neighbors need to be added
    neighbors = []
    if p[0] > 0:  # Add neighbours to list T, up
        a = (p[0] - 1, p[1])
        neighbors.append(a)
    if p[0] < img.shape[0] - 1:  # Add neighbours to list T, down
        b = (p[0] + 1, p[1])
        neighbors.append(b)
    if p[1] > 0:  # Add neighbours to list T, left
        c = (p[0], p[1] - 1)
        neighbors.append(c)
    if p[1] < img.shape[1] - 1:  # Add neighbours to list T, right
        d = (p[0], p[1] + 1)
        neighbors.append(d)
    return neighbors


def mean_region(img, reg):
    mean_value = []
    region_max = int(max(reg.flatten()))  # calculates amount of regions
    for count in range(1, region_max + 1):  # iterates over every region
        intensity = []
        for p in np.ndindex(img.shape):  # iterates over every pixel in the image
            if reg[p] == count:
                intensity.append(img[p])  # appends intensity value, if it is in the region
        mean_value.append(np.mean(intensity))  # calculates mean value of region
    return mean_value  # returns list with average of every region


if __name__ == '__main__':
    image = sk.imread("Data/N2DH-GOWT1/img/t01.tif")  # Bild laden
    img_s = image[300:400, 300:500]
    img_result = sd.seeds(img_s, 0.4, 40)
    img_result = sd.seed_merging(img_result)
    img_result = sd.decrease_region_number(img_result, 50)

    img_result = region_growing(img_s, img_result)
    ip.show_image(img_result, 15, 8)
