import numpy as np
from Functions import seeded_region_growing as srg


#  from Functions import image_processing as ip
#  from PIL import Image
#  import skimage.io as sk


def unseeded_calculate_one_distance(means, pixel_intensity, region_number):
    """
    calculates the distance of a pixel to the region with a specific region number
    :param means: list of means of the regions (list)
    :param pixel_intensity: the pixel which distance is calculated (tuple)
    :param region_number: number of a specific region (int)
    :return: the calculated distance for this pixel (float)
    """
    dist = np.abs(pixel_intensity - means[int(region_number) - 1])
    return dist


def unseeded_calculate_one_border_distances(means, img, max_region, one_border_neighbors):
    """
    calculates the distance of a pixel to the regions of the neighbors of one side
    :param means: list of means of the regions (list)
    :param img: array with intensity values (2D array)
    :param max_region: highest region number (int)
    :param one_border_neighbors: array with the region numbers of the neighbors to one side (2D array)
    :return: array with the calculated distances of the pixel to the region in the neighbors array (2D array)
    """
    one_border_distances = np.full(img.shape, 500)
    for region_number in range(1, max_region + 1):
        pos_reg_bor = np.where(one_border_neighbors == region_number)
        one_border_distances[pos_reg_bor[0], pos_reg_bor[1]] = \
            unseeded_calculate_one_distance(means, img[pos_reg_bor[0], pos_reg_bor[1]], region_number)
    return one_border_distances


def unseeded_calculate_distances(img, reg, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors):
    """
    creates four arrays with the distances of the pixels to their neighboring regions to all sides
    :param img: array with intensity values (2D array)
    :param reg: array with region numbers (2D array)
    :param left_neighbors: array with the region numbers of the neighbors to the left side (2D array)
    :param right_neighbors: array with the region numbers of the neighbors to the right side (2D array)
    :param top_neighbors: array with the region numbers of the neighbors to the upper side (2D array)
    :param bottom_neighbors: array with the region numbers of the neighbors to lower side (2D array)
    :return: means: list of means (list), and four arrays with the calculated distances to all the sides (2D arrays)
    """
    means = np.asarray(srg.mean_region(img, reg))
    max_region = int(np.amax(reg))

    left_distances = unseeded_calculate_one_border_distances(means, img, max_region, left_neighbors)
    right_distances = unseeded_calculate_one_border_distances(means, img, max_region, right_neighbors)
    top_distances = unseeded_calculate_one_border_distances(means, img, max_region, top_neighbors)
    bottom_distances = unseeded_calculate_one_border_distances(means, img, max_region, bottom_neighbors)

    return means, left_distances, right_distances, top_distances, bottom_distances


def unseeded_update_one_distance(img, reg, means, new_pixel, one_border_neighbors, one_border_distances):
    """
    updates the distance array of one direction
    :param img: array with intensity values (2D array)
    :param reg: array with region numbers (2D array)
    :param means: list of means of the regions (list)
    :param new_pixel: pixel which was lastly added to a region
    :param one_border_neighbors: array with the region numbers of the neighbors to one side (2D array)
    :param one_border_distances: array with the calculated distances of the pixel to one side (2D array)
    :return: updated array with the calculated distances of the pixel to one side (2D array)
    """
    positions_to_update = np.where(one_border_neighbors == reg[new_pixel])
    one_border_distances[positions_to_update[0],
                         positions_to_update[1]] = unseeded_calculate_one_distance(means, img[positions_to_update[0],
                                                                                              positions_to_update[1]],
                                                                                   reg[new_pixel])
    one_border_distances[new_pixel] = 500
    return one_border_distances


def unseeded_update_distances(img, reg, means, new_pixel, left_neighbors, right_neighbors, top_neighbors,
                              bottom_neighbors, left_distances, right_distances, top_distances, bottom_distances):
    """
    updates the list of means and the four distance arrays for all directions
    :param img: array with intensity values (2D array)
    :param reg: array with region numbers (2D array)
    :param means: list of means of the regions (list)
    :param new_pixel: pixel which was lastly added to a region
    :param left_neighbors: array with the region numbers of the neighbors to the left side (2D array)
    :param right_neighbors: array with the region numbers of the neighbors to the right side (2D array)
    :param top_neighbors: array with the region numbers of the neighbors to the upper side (2D array)
    :param bottom_neighbors: array with the region numbers of the neighbors to lower side (2D array)
    :param left_distances: array with the calculated distances of the pixel to region of left neighbors (2D array)
    :param right_distances: array with the calculated distances of the pixel to region of right neighbors (2D array)
    :param top_distances: array with the calculated distances of the pixel to region of upper neighbors (2D array)
    :param bottom_distances: array with the calculated distances of the pixel to region of lower neighbors (2D array)
    :return: four updated distance arrays (2D arrays), means: list of means (list)
    """
    means = srg.update_list_of_means(means, img, reg, new_pixel)

    left_distances = unseeded_update_one_distance(img, reg, means, new_pixel, left_neighbors, left_distances)
    right_distances = unseeded_update_one_distance(img, reg, means, new_pixel, right_neighbors, right_distances)
    top_distances = unseeded_update_one_distance(img, reg, means, new_pixel, top_neighbors, top_distances)
    bottom_distances = unseeded_update_one_distance(img, reg, means, new_pixel, bottom_neighbors,
                                                    bottom_distances)

    return left_distances, right_distances, top_distances, bottom_distances, means


def unseeded_choose_distance_array(border_number, left_distances, right_distances, top_distances, bottom_distances):
    """
    chooses one of the four distance arrays by the number of the array
    :param border_number: number of the array which should be used (int)
    :param left_distances: array with the calculated distances of the pixel to region of left neighbors (2D array)
    :param right_distances: array with the calculated distances of the pixel to region of right neighbors (2D array)
    :param top_distances: array with the calculated distances of the pixel to region of upper neighbors (2D array)
    :param bottom_distances: array with the calculated distances of the pixel to region of lower neighbors (2D array)
    :return: name of the distance array which should be used (name of variable)
    """
    if border_number == 0:
        return left_distances
    elif border_number == 1:
        return right_distances
    elif border_number == 2:
        return top_distances
    return bottom_distances


def unseeded_label_new_pixel(reg, left_distances, right_distances, top_distances, bottom_distances, left_neighbors,
                             right_neighbors, top_neighbors, bottom_neighbors, t, means, img):
    """
    determines the pixel with the smallest distance and decides how it should be labeled to a neighboring region,
    another region or should be a new region and updates the neighbor arrays for all directions
    :param reg: array with region numbers (2D array)
    :param left_distances: array with the calculated distances of the pixel to region of left neighbors (2D array)
    :param right_distances: array with the calculated distances of the pixel to region of right neighbors (2D array)
    :param top_distances: array with the calculated distances of the pixel to region of upper neighbors (2D array)
    :param bottom_distances: array with the calculated distances of the pixel to region of lower neighbors (2D array)
    :param left_neighbors: array with the region numbers of the neighbors to the left side (2D array)
    :param right_neighbors: array with the region numbers of the neighbors to the right side (2D array)
    :param top_neighbors: array with the region numbers of the neighbors to the upper side (2D array)
    :param bottom_neighbors: array with the region numbers of the neighbors to lower side (2D array)
    :param t: threshold to decide whether a pixel is similar enough to a region (float)
    :param means: list of means of the regions (list)
    :param img: array with intensity values (2D array)
    :return: reg: array with region numbers (2D array), pos_min_dist: pixel that was lastly added to a region (tuple),
    four arrays with region numbers of neighbors (2D arrays), means: list of means (list)
    """
    pos_min_dist, border_number = srg.position_of_smallest_distance(left_distances, right_distances, top_distances,
                                                                    bottom_distances)
    one_border_neighbors = srg.choose_border_number(border_number, left_neighbors, right_neighbors, top_neighbors,
                                                    bottom_neighbors)
    one_border_distances = unseeded_choose_distance_array(border_number, left_distances, right_distances, top_distances,
                                                          bottom_distances)

    if one_border_distances[pos_min_dist] < t:
        reg[pos_min_dist] = one_border_neighbors[pos_min_dist]
    else:
        distances = np.abs(img[pos_min_dist] - means)
        minimum = np.amin(distances)
        pos_minimum = np.where(distances == minimum)[0]
        if minimum < t:
            reg[pos_min_dist] = pos_minimum[0] + 1
        else:
            region_max = int(max(reg.flatten()))
            reg[pos_min_dist] = region_max + 1
            means = np.append(means, img[pos_min_dist])

    left_neighbors = srg.update_left_neighbors(reg, left_neighbors, pos_min_dist)
    right_neighbors = srg.update_right_neighbors(reg, right_neighbors, pos_min_dist)
    top_neighbors = srg.update_top_neighbors(reg, top_neighbors, pos_min_dist)
    bottom_neighbors = srg.update_bottom_neighbors(reg, bottom_neighbors, pos_min_dist)

    return reg, pos_min_dist, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors, means


def unseeded_region_growing_algorithm(img, start_pixel, t):
    """
    performs the unseeded region growing algorithm on an image, with a set region array with a startpixel and
    a threshold to decide whether a pixel is similar enough to be added to a certain region
    :param start_pixel: pixel which defines the starting seed (tuple)
    :param img: array with intensity values (2D array)
    :param t: threshold to decide whether a pixel is similar enough to a region (float)
    :return: array with region numbers (2D array)
    """
    reg = np.zeros(img.shape, int)  # array with region number
    reg[start_pixel] = 1
    size = img.shape[0] * img.shape[1]

    left_neighbors, right_neighbors, top_neighbors, bottom_neighbors = srg.find_seed_neighbors(reg)

    means, left_distances, right_distances, top_distances, bottom_distances = \
        unseeded_calculate_distances(img, reg, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors)

    reg, pos_min_dist, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors, means = \
        unseeded_label_new_pixel(reg, left_distances, right_distances, top_distances, bottom_distances, left_neighbors,
                                 right_neighbors, top_neighbors, bottom_neighbors, t, means, img)

    i = 0

    while srg.unlabeled_pixel_exist(reg):

        i += 1
        if i % 5000 == 0:
            print(i)

        left_distances, right_distances, top_distances, bottom_distances, means = \
            unseeded_update_distances(img, reg, means, pos_min_dist, left_neighbors, right_neighbors, top_neighbors,
                                      bottom_neighbors, left_distances, right_distances, top_distances,
                                      bottom_distances)

        reg, pos_min_dist, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors, means = \
            unseeded_label_new_pixel(reg, left_distances, right_distances, top_distances, bottom_distances,
                                     left_neighbors, right_neighbors, top_neighbors, bottom_neighbors, t,
                                     means, img)

        #  print(np.count_nonzero(reg == 0))
    return reg
