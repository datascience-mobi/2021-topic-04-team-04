import skimage.io as sk
import numpy as np
from Functions import image_processing as ip
from Functions import seed_detection as sd


# import math as m
# import matplotlib.pyplot as plt

def find_neighbors(reg):
    """
    selects all 4 adjacent neighbors for every pixel
    :param reg: array with region numbers of the pixel (2D array)
    :return: list of all positions of neighboring pixels (list)
    """
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


def get_neighbors(img, p):
    """
    Finds maximal 4 direct neighbors of pixels, ignores border 
    :param img: image with intensity values (2D array)
    :param p: pixel for which the neighbors should be found (tuple with position)
    :return: list of maximal four neighbors (list)
    """  # p describes pixel for which neighbors need to be added
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
    """
    :param img: image with intensity values (2D array)
    :param reg: array with pixle numbers (2D array)
    :return: list with mean values of the regions, region number 1 has index 0 (list)
    """
    mean_value = []
    region_max = int(max(reg.flatten()))  # calculates amount of regions
    for count in range(1, region_max + 1):  # iterates over every region
        intensity = []
        for p in np.ndindex(img.shape):  # iterates over every pixel in the image
            if reg[p] == count:
                intensity.append(img[p])  # appends intensity value, if it is in the region
        mean_value.append(np.mean(intensity))  # calculates mean value of region
    return mean_value  # returns list with average of every region


def one_region_mean(img, reg, new_pixel):
    """
    recalculates mean value of region with added pixel
    :param img: intensity values of image (2d array)
    :param reg: region numbers of image pixels (2d array)
    :param new_pixel: position of newly labeled pixel (tuple)
    :return: updated mean value of changed region (float)
    """
    pos_new_reg = np.where(reg == reg[new_pixel])
    single_mean = np.mean(img[pos_new_reg[0], pos_new_reg[1]])
    return single_mean  # returns mean value of changed region


def calculation_distance(img, neighbors, reg):
    """
    calculates distances of all
    :param img: image, intensity values of pixels (2d array)
    :param neighbors: list of positions (tuples (x,y)) of neighboring pixels which are to be labeled next
    :param reg: region numbers (int) of image pixels (2d array) (0 if pixel unlabeled)
    :return: distances: minimal distances of to be labeled pixels (neighbors) (2d array, one if already labeled or not
             to be labeled, values between 0 and 1)
    :return: nearest_reg: numbers of nearest region (2d array)
    :return: list of mean values of regions(list)
    """
    max_intensity = np.amax(img)
    means = mean_region(img, reg)
    distances = np.ones(img.shape)
    nearest_reg = np.zeros(img.shape)

    for pixel in neighbors:
        four_neighbors = get_neighbors(img, pixel)

        distance = []
        region_number = []

        for neighbor_position in four_neighbors:
            if is_labeled(reg, neighbor_position):
                distance.append(calculate_distance(img, means, max_intensity, pixel, reg[neighbor_position]))
                region_number.append(reg[neighbor_position])
        min_dist = min(distance)
        pos_min_dist = distance.index(min(distance))
        nearest_reg[pixel] = region_number[int(pos_min_dist)]
        distances[pixel] = min_dist
    return distances, nearest_reg, means


def is_labeled(reg, position):
    """
    tests whether a pixel already has a region
    :param reg: array with region numbers (2D array)
    :param position:
    :return: True/False
    """
    if reg[position] != 0:
        return True
    return False


def calculate_distance(img, means, max_intensity, pixel, neighboring_region):
    """
    calculates distance of pixel to neighboring region
    :param img: intensity values (2d array)
    :param means: list of mean values to region numbers (list)
    :param max_intensity: maximal intensity of image (float)
    :param pixel: position of pixel for witch distance is calculated (tuple (x,y))
    :param neighboring_region: neighboring region to which distance is calculated (int)
    :return: distance
    """
    distance = np.abs((img[pixel] - means[int(neighboring_region) - 1])) / max_intensity
    return distance


def new_distance(img, reg, nearest_reg, dis, new_pixel, neighbors, means):
    """"
    updates distances of to be labeled pixels to neighboring regions
    :param img: intensity values (2d array)
    :param reg: region numbers (2d array)
    :param nearest_reg: numbers of nearest regions(2d array)
    :param dis: distances to nearest region (2d array)
    :param new_pixel: position of newly labeled pixel (tuple (x,y))
    :param neighbors: list of pixels to be sorted (list of tuples(x,y))
    :param means: list of mean intensity values of regions (list of floats)

    :return: dist: updated minimal distances of to be labeled pixels (neighbors) (2d array, one if already labeled or
             not to be labeled, values between 0 and 1)
    :return: nearest_reg: updated numbers of nearest region (2d array)
    :return: list of mean intensity values of regions(list)
    """
    means = update_list_of_means(means, img, reg, new_pixel)
    max_intensity = np.amax(img)

    for pixel in neighbors:
        four_neighbors = get_neighbors(img, pixel)  # list 4 neighbors of pixel i out of unsorted neighbors list

        distance = []
        region_number = []
        recalculated = False

        for neighbor_position_to_change in four_neighbors:
            if not recalculated:
                if pixel_in_new_region(reg, neighbor_position_to_change, new_pixel):
                    recalculated = True
                    for neighbor_position in four_neighbors:
                        if is_labeled(reg, neighbor_position):
                            distance.append(calculate_distance(img, means, max_intensity, pixel, reg[neighbor_position]))
                            region_number.append(reg[neighbor_position])

                    min_dist = min(distance)
                    pos_min_dist = distance.index(min_dist)
                    nearest_reg[pixel] = region_number[
                        int(pos_min_dist)]
                    dis[pixel] = min_dist
    return dis, nearest_reg, means


def pixel_in_new_region(reg, neighbor_position, new_pixel):
    """
    decides whether the region on the neighbor pixel is the region of the previously assigned pixel
    :param reg: array with region number (2D array)
    :param neighbor_position: position of the neighbor pixel (tuple)
    :param new_pixel: pixel which was just added to a region (tuple)
    :return: True/False
    """
    if reg[neighbor_position] == reg[new_pixel]:
        return True
    return False


def update_list_of_means(means, img, reg, new_pixel):
    """
    updates the mean list
    :param means: list of means which needs to be updated (list)
    :param img: array with intensity values (2D array)
    :param reg: array with region numbers (2D array)
    :param new_pixel: pixel which was just added to a region (tuple)
    :return: updated list of means (list)
    """
    new_mean = one_region_mean(img, reg, new_pixel)
    means[int(reg[new_pixel] - 1)] = new_mean
    return means


def label(reg, dis, nearest_reg, neighbors):
    """
    labels one pixel (nearest pixel to one of the regions)
    :param reg: region numbers (2d array of ints)
    :param dis: distances to nearest neighboring region (2d array)
    :param nearest_reg: number of nearest neighboring region (2d array)
    :param neighbors: list of to be labeled pixels (list of tuples (x,y))

    :return: reg: updated regions of pixels (2d array)
    :return: pos_min_dist: position of newly labeled pixel (tuple (x,y))
    :return: neighbors: updated list of pixels to be labeled
    :return: dis: updated distances to nearest region (2d array)
    """
    pos_min_dist = position_of_smallest_distance(dis)
    reg[pos_min_dist] = nearest_reg[pos_min_dist]
    neighbors.remove(pos_min_dist)
    dis[pos_min_dist] = 1
    #print(len(neighbors))
    return reg, pos_min_dist, neighbors, dis


def position_of_smallest_distance(dis):
    """
    picks the pixel with the smallest distance
    :param dis: array with minimal distances of the pixel (2D array)
    :return: pixel with the smallest distance (tuple)
    """
    minimal_distances = np.where(dis == np.amin(dis))
    pos_min_dist = list(zip(minimal_distances[0], minimal_distances[1]))[0]
    pos_min_dist = (int(pos_min_dist[0]), int(pos_min_dist[1]))
    return pos_min_dist


def region_growing(img, reg):
    """
    performs region growing algorithm on image with defined seeds (reg)
    :param img: intensity values (2d array)
    :param reg: region numbers, predefined from seed selection (2d array)
    :return: regions of all pixels, result of seeded region growing (2d array), values start with 1 (ints)
    """

    neighbors = find_neighbors(reg)
    dist = calculation_distance(img, neighbors, reg)

    regions_new = label(reg, dist[0], dist[1], neighbors)
    neighbors = regions_new[2]
    distances = regions_new[3]
    neighbors = add_missing_neighbors(img, regions_new[1], neighbors, reg)

    i = 0
    while unlabeled_pixel_exist(neighbors):
        i += 1
        if i % 1000 == 0:
            print(i)

        dist = new_distance(img, regions_new[0], dist[1], distances, regions_new[1], neighbors,
                            dist[2])
        regions_new = label(regions_new[0], dist[0], dist[1], neighbors)  # labels next pixel
        neighbors = regions_new[2]
        distances = regions_new[3]
        neighbors = add_missing_neighbors(img, regions_new[1], neighbors, reg)

    return regions_new[0]


def unlabeled_pixel_exist(neighbors):
    """
    decides, whether an pixel without region exists
    :param neighbors: list with neighbor pixels (list)
    :return: True/False
    """
    if len(neighbors) > 0:
        return True
    return False


def add_missing_neighbors(img, pos_min_dist, neighbors, reg):
    """
    adds missing neighbors to the neighbors list
    :param img: array with intensity values (2D array)
    :param pos_min_dist: pixel with the smallest distance (tuple)
    :param neighbors: old neighbors list (list)
    :param reg: array with region numbers (2D array)
    :return: list containing all the neighbors (list)
    """
    neighbors_to_add = get_neighbors(img, pos_min_dist)  # finds neighbors of newly labeled pixel
    for neighbor in neighbors_to_add:
        if neighbor not in neighbors and not is_labeled(reg, neighbor):
            neighbors.append(neighbor)
    return neighbors


if __name__ == '__main__':
    image = sk.imread("../Data/N2DH-GOWT1/img/t01.tif")  # load image
    img_s = image[300:400, 300:500]
    img_result = sd.seeds(img_s, 40)
    img_result = sd.seed_merging(img_result)
    #img_result = sd.decrease_region_number(img_result, 50)

    img_result = region_growing(img_s, img_result)
    ip.show_image(img_result, 15, 8)
