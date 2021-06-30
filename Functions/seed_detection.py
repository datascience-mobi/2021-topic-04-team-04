import numpy as np
import math as m
from collections import Counter
from Functions import image_processing as ip
from skimage.filters import threshold_otsu

# import skimage.io as sk


def standard_deviation(img, size):
    """  calculates standard deviation of every pixel
    :param img: intensity values of the image (2d array)
    :param size: size of the filter mask (int)
    :return: array with standard deviation of every pixel (2d array)
    """
    result = np.zeros(img.shape)  # create empty array (zeros)
    n = (size - 1) // 2
    for p in np.ndindex(img.shape):  # iterates over every pixel
        neighborhood_sum = 0
        deviation = 0
        if not ip.is_border_pixel(p, img):
            for q in np.ndindex(size, size):  # iterates over filter mask (nxn neighborhood)
                i = p[0] - n + q[0]
                j = p[1] - n + q[1]
                neighborhood_sum += img[i, j]
            mean = neighborhood_sum / (size ** 2)  # calculate mean in nxn neighborhood
            for q in np.ndindex(size, size):  # iterates over filter mask (nxn neighborhood)
                i = p[0] - n + q[0]
                j = p[1] - n + q[1]
                deviation += (img[i, j] - mean) ** 2  # calculate deviation from mean

            result[p] = m.sqrt(1 / (size ** 2) * deviation)  # calculate standard deviation
    return result


def euclidean_relative(img, size):
    """ calculates maximum relative euclidean distance for every pixel in nxn neighborhood
    :param img: intensity values of image (2d array)
    :param size: size of the filter mask/ neighborhood (int)
    :return: array with maximal euclidean distance of every pixel (2d array)
    """
    result = np.zeros(img.shape)
    n = (size - 1) // 2
    for p in np.ndindex(img.shape):
        neighborhood_distance = []
        if not ip.is_border_pixel(p, img):
            for q in np.ndindex(size, size):
                i = p[0] - n + q[0]
                j = p[1] - n + q[1]
                neighborhood_distance.append((img[p] - img[i, j]) / img[p])  # adds relative euclidean distance to list
            result[p] = max(neighborhood_distance)  # chooses maximum distance
    return result


def euclidean_n(img, size):
    """ # calculates maximum euclidean distance for every pixel in nxn neighborhood
    :param img: intensity values of image (2d array)
    :param size: size of the filter mask/ neighborhood (int)
    :return: array with maximal, relative euclidean distance of every pixel (2d array)
    """
    result = np.zeros(img.shape)
    n = (size - 1) // 2
    for p in np.ndindex(img.shape):
        neighborhood_distance = []
        if not ip.is_border_pixel(p, img):
            for q in np.ndindex(size, size):
                i = p[0] - n + q[0]
                j = p[1] - n + q[1]
                neighborhood_distance.append((img[p] - img[i, j]))
            result[p] = max(neighborhood_distance)
    return result


def otsu_thresholding(img):
    """
    defines the optimal otsu-threshold for the image
    :param img:
    :return: threshold for seed detection criteria similarity (float)
    """
    otsu_threshold = threshold_otsu(img)
    return otsu_threshold


def seeds(img, threshold_similarity, threshold_distance):
    """ automatic seed selection algorithm
    :param img: intensity values of image (2d array)
    :param threshold_similarity: threshold for the similarity, calculated with otsu-method (float)
    :param threshold_distance: Threshold for the relative euclidean distance
    :return: all seeds get intensity value 1 every other pixel has intensity value 0 (2d array)
             every border pixel is here detected as a seed but will be removed in seed merging
    """
    result = np.zeros(img.shape)
    sd_seeds = standard_deviation(img, 3)
    sd_flat = sd_seeds.flatten()
    similarity_seeds = 1 - sd_seeds / max(sd_flat)  # calculates similarity of every pixel to its neighbors
    relative_euclidean_distance_seeds = euclidean_relative(img, 3)
    for p in np.ndindex(img.shape):  # border pixel value is zero
        if similarity_seeds[p] > threshold_similarity and relative_euclidean_distance_seeds[p] < threshold_distance:
            result[p] = 1
    return result


def seed_merging(img):
    """
    :param img: array where every seed has the value 1 and every other pixel the value 0 (2d array)
    :return: array with merged seeds (2d array)
    """
    regions = np.zeros(img.shape)  # creates new array for region numbers for every image pixel
    count = 1  # keep track of region number
    for p in np.ndindex(img.shape):  # iterates over every pixel of the image
        if img[p] == 1:  # tests if pixel is seed
            if not ip.is_border_pixel(p, img):  # no calculation of border pixels
                for q in np.ndindex(3, 3):  # iterates over 3x3 neighborhood
                    i = p[0] - 1 + q[0]
                    j = p[1] - 1 + q[1]
                    if regions[i, j] != 0:  # tests if neighbors are also seeds
                        regions[p] = regions[i, j]  # merge neighboring seeds; Zuordnung zur letzten Region
                if regions[p] == 0:  # tests if no neighbors are seeds
                    regions[p] = count  # creates new region from new seed
                    count += 1
    return regions


def decrease_region_number(img, threshold):
    """ reduce number of starting regions for region growing, only considering regions with more than T seeds
    :param img: array with merged seeds (2d array)
    :param threshold: Threshold für die size of seed regions (int)
    :return: array with big seeds (2d array)
    """
    count = Counter(img.flatten())  # counts number of seeds in region
    d_seeds = img.copy()
    for i in range(1, int(np.amax(img))):  # iterates over every region
        if count[i] <= threshold:  # if number of seeds is smaller than threshold, delete region
            for p in np.ndindex(img.shape):
                if img[p] == i:
                    d_seeds[p] = 0
    return d_seeds
