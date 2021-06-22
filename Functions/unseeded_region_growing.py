import numpy as np
from Functions import old_seeded_region_growing as srg

# import skimage.io as sk
# import matplotlib.pyplot as plt
# import math as m
# from Functions import image_processing as ip


def unseeded_distance(img, neighbors, reg):
    """
    calculates intensity distance from unsorted neighbor pixel to mean intensity of regions ?
    :param img: image with intensity values (2D array)
    :param neighbors: list of neighbors of the regions (list)
    :param reg: array with regions (2D array)
    :return: distance array, all non neighboring pixels have value 500 (2D array)
             and number of region with closest intensity distance from the pixel (int)
    """
    means = srg.mean_region(img, reg)
    distance = np.full(img.shape, 500)  # value 500 so all calculated distances are smaller
    nearest_region = np.zeros(img.shape)
    for i in neighbors:
        pixel_neighbors = srg.get_neighbors(img, i)
        distance_list = []
        region_number = []
        for j in pixel_neighbors:
            if reg[j] != 0:  # neighbors with assigned regions
                distance_list.append(abs(img[i] - means[int(reg[j] - 1)]))  # calculates intensity distance
                region_number.append(reg[j])
        distance[i] = min(distance_list)  # minimal distance written in array
        nearest_region[i] = region_number[distance_list.index(min(distance_list))]
    return distance, nearest_region


def unseeded_pixel_pick(dis):
    """
    selects pixel form neighbor list with minimal intensity distance
    :param dis: array with intensity distances (2D array)
    :return: position of the pixel with minimal intensity distance (tuple (x,y) with position in an array)
    """
    x = np.where(dis == np.amin(dis))
    minimum = list(zip(x[0], x[1]))[0]
    pick = (int(minimum[0]), int(minimum[1]))
    return pick


def unseeded_region_direct(reg, pick, nearest_region): #treshold: Pick soll nur zugeordnet werden, wenn Distanz kleiner t ist
    """
    assigns pick to the region region with minimal intensity distance
    :param nearest_region: array with number of the region with smallest distance (2D array)
    :param reg: array with region numbers (2D array)
    :param pick: selected pixel (tuple, position in an array)
    :return: array with updated regions (2D array)
    """
    reg[pick] = nearest_region[pick]
    return reg


def unseeded_region_indirect_or_new(img, reg, pick, threshold):
    """
    calculates distance to all regions and assigns it to the smallest intensity distance or creates new region
    :param img: image with intensity values (2D array)
    :param reg: array with region numbers (2D array)
    :param pick: selected pixel (tuple)
    :param threshold: Threshold for decision if pixel is in new region (int)
    :return: array with updated regions (2D array)
    """
    means = srg.mean_region(img, reg)
    distance_list = []
    for m in means:
        distance_list.append(abs(img[pick] - m))  # calculates all distances to the regions
    minimum = min(distance_list)  # selects minimal distance
    if minimum < threshold:
        reg[pick] = distance_list.index(minimum) + 1  # first region with minimal value
    else:
        region_max = int(max(reg.flatten()))  # highest region number
        reg[pick] = region_max + 1  # new region number for pick
    return reg
