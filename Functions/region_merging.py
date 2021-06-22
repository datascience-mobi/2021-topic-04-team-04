import numpy as np
from Functions import seeded_region_growing as srg


#  from Functions import image_processing as ip
#  from Functions import seed_detection as sd
#  import skimage.io as sk


def region_distance(img, reg):
    """
    calculates distance between all regions
    :param img: intensity values (2d array)
    :param reg: region numbers (2d array)
    :return: 2d array with distances between all regions, only in upper pyramid
    (region numbers are row and col number)
    mean values of all regions (list of floats)
    """
    max_region = max(reg.flatten())
    inter_region_distances = np.ones((int(max_region), int(max_region)))
    means = srg.mean_region(img, reg)
    for row_number in range(0, int(max_region)):
        for col_number in range(0, int(max_region)):
            if col_number > row_number:
                distance = abs(means[row_number] - means[col_number]) / (np.amax(img))
                inter_region_distances[row_number][col_number] = distance
    return inter_region_distances, means


def one_merged_region_mean(img, reg, region_number):
    """
    calculates the mean of changed region
    :param img: intensity values (2d array)
    :param reg: region numbers (2d array)
    :param region_number: number of the changed region (int)
    :return: mean value (float)
    """
    pos_new_reg = np.where(reg == region_number)
    single_mean = np.mean(img[pos_new_reg[0], pos_new_reg[1]])
    return single_mean  # returns mean value of changed region


def region_distance_new(img, reg, pos_min_dist, means, inter_region_distances):
    region_count = inter_region_distances.shape[0]
    maximal_intensity = np.amax(img)
    changed_region1 = int(pos_min_dist[0])  # Number of first region is used
    changed_region2 = int(pos_min_dist[1])
    means = update_mean_values(means, changed_region1, changed_region2, img, reg)
    inter_region_distances = update_distances(changed_region1, changed_region2, inter_region_distances, region_count,
                                              means, maximal_intensity)
    return inter_region_distances


def update_distances(changed_region1, changed_region2, inter_region_distances, region_count, means, maximal_intensity):
    for col_number in range(changed_region1 + 1, int(region_count)):  # Number of second region, not longer used
        distance = abs(means[changed_region1] - means[col_number]) / maximal_intensity
        inter_region_distances[changed_region1][col_number] = distance
    for row_number in range(0, changed_region1):
        distance = abs(means[changed_region1] - means[row_number]) / maximal_intensity
        inter_region_distances[row_number][changed_region1] = distance
    inter_region_distances[changed_region2][0:region_count] = 500
    inter_region_distances[0:region_count][changed_region2] = 500
    return inter_region_distances


def update_mean_values(means, changed_region1, changed_region2, img, reg):
    means[changed_region2] = 500
    means[changed_region1] = one_merged_region_mean(img, reg, changed_region1 + 1)
    return means


def updates_region_numbers(inter_region_distances, reg, min_distance):
    x = np.where(inter_region_distances == min_distance)  # finds minimal distance in array and its position
    pos_min_dist = list(zip(x[0], x[1]))[0]  # position of first pixel with minimal distance value
    pos_min_dist = (int(pos_min_dist[0]), int(pos_min_dist[1]))  # converting to int for finding in list
    pixel_to_change = np.where(reg == pos_min_dist[1] + 1)
    # region to be changed is always the column in inter_region_distance (column number always bigger than row number)
    pixel_to_change_rows = pixel_to_change[0]
    pixel_to_change_cols = pixel_to_change[1]
    reg[pixel_to_change_rows, pixel_to_change_cols] = pos_min_dist[0] + 1
    return reg, pos_min_dist


def distance_merging_while(reg, t, img):
    result_region_distance = region_distance(img, reg)
    inter_region_distances = result_region_distance[0]
    means = result_region_distance[1]
    min_distance = np.nanmin(inter_region_distances)
    #  print(min_distance)
    while min_distance < t:
        #  print(1)
        updated_regions = updates_region_numbers(inter_region_distances, reg, min_distance)
        reg = updated_regions[0]
        pos_min_dist = updated_regions[1]
        inter_region_distances = region_distance_new(img, reg, pos_min_dist, means, inter_region_distances)
        min_distance = np.nanmin(inter_region_distances)
        #  print(min_distance)
        #  print(inter_region_distances)
    return reg
