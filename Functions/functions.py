import numpy as np
import math as m

# import skimage.io as sk
# import matplotlib.pyplot as plt

def standarddeviation(img, size):  # calculates standard deviation of every pixel (image, size of filter mask)
    result = np.zeros(img.shape)  # create empty array (zeros)
    n = (size - 1) // 2
    for p in np.ndindex(img.shape):  # iterates over every pixel
        neighborhood_sum = 0
        mean = 0
        deviation = 0
        if p[0] - n >= 0 and p[1] - n >= 0 and p[0] + n <= img.shape[0] - 1 and p[1] + n <= img.shape[
            1] - 1:  # no calculation of border pixels
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


def euclidean_relative(img, size):  # calculates maximum relative euclidean distance for every pixel in nxn neighborhood
    result = np.zeros(img.shape)  # create empty array (zeros)
    n = (size - 1) // 2
    for p in np.ndindex(img.shape):  # iterates over every pixel using a touple
        neighborhood_distance = []  # creates a list for distances
        if p[0] - n >= 0 and p[1] - n >= 0 and p[0] + n <= img.shape[0] - 1 and p[1] + n <= img.shape[
            1] - 1:  # no calculation of border pixels
            for q in np.ndindex(size, size):  # iterates over nxn neighborhood
                i = p[0] - n + q[0]
                j = p[1] - n + q[1]
                neighborhood_distance.append((img[p] - img[i, j]) / img[p])  # adds relative euclidean distance to list

            result[p] = max(neighborhood_distance)  # chooses maximum distance
    return result


def euclidean_n(img, size):  # calculates maximum euclidean distance for every pixel in nxn neighborhood
    result = np.zeros(img.shape)  # create empty array (zeros)
    n = (size - 1) // 2
    for p in np.ndindex(img.shape):  # iterates over every pixel using a touple
        neighborhood_distance = []  # creates a list for distances
        if p[0] - n >= 0 and p[1] - n >= 0 and p[0] + n <= img.shape[0] - 1 and p[1] + n <= img.shape[
            1] - 1:  # no calculation of border pixels
            for q in np.ndindex(size, size):  # iterates over nxn neighborhood
                i = p[0] - n + q[0]
                j = p[1] - n + q[1]
                neighborhood_distance.append((img[p] - img[i, j]))  # adds euclidean distance to list
            result[p] = max(neighborhood_distance)  # chooses maximum distance
    return result
