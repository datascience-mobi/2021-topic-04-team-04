import numpy as np
import math as m
from Functions import image_processing as ip
from skimage.filters import threshold_otsu


def standard_deviation(img, size):
    """  calculates standard deviation of every pixel
    :param img: intensity values of the image (2d array)
    :param size: size of the filter mask (int)
    :return: array with standard deviation of every pixel (2d array)
    """
    standard_dev = np.zeros(img.shape)
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

            standard_dev[p] = m.sqrt(1 / (size ** 2) * deviation)  # calculate standard deviation
    return standard_dev


def euclidean_relative(img, size):
    """ calculates maximum relative euclidean distance for every pixel in nxn neighborhood
    :param img: intensity values of image (2d array)
    :param size: size of the filter mask/ neighborhood (int)
    :return: array with maximal euclidean distance of every pixel (2d array)
    """
    maximal_euclidean_distance = np.zeros(img.shape)
    n = (size - 1) // 2
    for p in np.ndindex(img.shape):
        neighborhood_distance = []
        if not ip.is_border_pixel(p, img):
            for q in np.ndindex(size, size):
                i = p[0] - n + q[0]
                j = p[1] - n + q[1]
                neighborhood_distance.append((np.float(img[p]) - np.float(img[i, j])) / (np.float(img[p]) + 0.00000001))
                # adds relative euclidean distance to list, adds 0.000001 to prevent division by 0
            maximal_euclidean_distance[p] = max(neighborhood_distance)  # chooses maximum distance
    return maximal_euclidean_distance


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
    :param img: intensity values of the image (2d array)
    :return: threshold for seed detection criteria similarity (float)
    """
    otsu_threshold = threshold_otsu(img) / np.amax(img)
    #  otsu_threshold = 0.1
    #  print(otsu_threshold)
    return otsu_threshold


def seeds(img, threshold_distance):
    """ automatic seed selection algorithm
    :param img: intensity values of image (2d array)
    :param threshold_distance: Threshold for the relative euclidean distance
    :return: all seeds get intensity value 1 every other pixel has intensity value 0 (2d array)
             every border pixel is here detected as a seed but will be removed in seed merging
    """
    result = np.zeros(img.shape)
    threshold_similarity = otsu_thresholding(img)
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
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for pixel in np.ndindex(img.shape):  # iterates over every pixel of the image
        if img[pixel] == 1:  # tests if pixel is seed
            if not ip.is_border_pixel(pixel, img):  # no calculation of border pixels
                for element in neighbors:  # iterates over neighbors
                    neighbor = (pixel[0] + element[0], pixel[1] + element[1])
                    if regions[neighbor] != 0:  # tests if neighbors are also seeds
                        regions[pixel] = regions[neighbor]  # merge neighboring seeds; Zuordnung zur letzten Region
                if regions[pixel] == 0:  # tests if no neighbors are seeds
                    regions[pixel] = count  # creates new region from new seed
                    count += 1
    return regions


def reduce_region_number(reg, threshold):
    """
    reduces number of seeds
    :param reg: region numbers (2d array)
    :param threshold: seed regions smaller than threshold are removed (int)
    :return: region numbers (2d array)
    """
    region_count = int(np.max(reg))
    counter = 1
    for region_number in range(1, region_count + 1):
        pos_of_reg = np.where(reg == region_number)
        size = len(pos_of_reg[0])
        if size >= threshold:
            reg[pos_of_reg[0], pos_of_reg[1]] = counter
            counter += 1
        else:
            reg[pos_of_reg[0], pos_of_reg[1]] = 0
    return reg
