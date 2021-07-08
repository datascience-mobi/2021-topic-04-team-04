import skimage.io as sk
import numpy as np
from Functions import image_processing as ip
from Functions import seed_detection as sd
from PIL import Image


def calculate_left_neighbors(reg, seeds):
    """
    determines the region number of the left neighbor pixel of every seed
    :param reg: array with region numbers (2D array)
    :param seeds: array where every seed has the value 1 and every other pixel the value 0 (2D array)
    :return: array saving the region number of the left pixel of every seed (2D array)
    """
    left_neighbors = np.zeros((reg.shape[0], reg.shape[1] + 1))
    left_neighbors[seeds[0], seeds[1]] = reg[seeds[0], seeds[1]]
    left_neighbors = np.delete(left_neighbors, 0, axis=1)
    left_neighbors[seeds[0], seeds[1]] = 0  # remove already labeled pixels
    return left_neighbors


def calculate_right_neighbors(reg, seeds):
    """
    determines the region number of the right neighbor pixel of every seed
    :param reg: array with region numbers (2D array)
    :param seeds: array where every seed has the value 1 and every other pixel the value 0 (2D array)
    :return: array saving the region number of the right pixel of every seed (2D array)
    """
    right_neighbors = np.zeros((reg.shape[0], reg.shape[1] + 1))
    right_neighbors[seeds[0], seeds[1] + 1] = reg[seeds[0], seeds[1]]
    right_neighbors = np.delete(right_neighbors, reg.shape[1] - 1, axis=1)
    right_neighbors[seeds[0], seeds[1]] = 0
    return right_neighbors


def calculate_top_neighbors(reg, seeds):
    """
    determines the region number of the upper neighbor pixel of every seed
    :param reg: array with region numbers (2D array)
    :param seeds: array where every seed has the value 1 and every other pixel the value 0 (2D array)
    :return: array saving the region number of the upper pixel of every seed (2D array)
    """
    top_neighbors = np.zeros((reg.shape[0] + 1, reg.shape[1]))
    top_neighbors[seeds[0], seeds[1]] = reg[seeds[0], seeds[1]]
    top_neighbors = np.delete(top_neighbors, 0, axis=0)
    top_neighbors[seeds[0], seeds[1]] = 0
    return top_neighbors


def calculate_bottom_neighbors(reg, seeds):
    """
    determines the region number of the lower neighbor pixel of every seed
    :param reg: array with region numbers (2D array)
    :param seeds: array where every seed has the value 1 and every other pixel the value 0 (2D array)
    :return: array saving the region number of the lower pixel of every seed (2D array)
    """
    bottom_neighbors = np.zeros((reg.shape[0] + 1, reg.shape[1]))
    bottom_neighbors[seeds[0] + 1, seeds[1]] = reg[seeds[0], seeds[1]]
    bottom_neighbors = np.delete(bottom_neighbors, reg.shape[0] - 1, axis=0)
    bottom_neighbors[seeds[0], seeds[1]] = 0
    return bottom_neighbors


def find_seed_neighbors(reg):
    """
    creates four arrays with the region numbers of the neighbors in all directions
    :param reg: array with region numbers (2D array)
    :return: four arrays with the region numbers of the neighbors in one direction (2D arrays)
    """
    seeds = np.where(reg != 0)

    left_neighbors = calculate_left_neighbors(reg, seeds).astype(int)
    right_neighbors = calculate_right_neighbors(reg, seeds).astype(int)
    top_neighbors = calculate_top_neighbors(reg, seeds).astype(int)
    bottom_neighbors = calculate_bottom_neighbors(reg, seeds).astype(int)

    return left_neighbors, right_neighbors, top_neighbors, bottom_neighbors


def mean_region(img, reg):
    """
    calculates mean intensity value of every region
    :param img: intensity values (2D array)
    :param reg: array with region numbers (2D array)
    :return: list with mean values of the regions, region number 1 has index 0 (list)
    """
    mean_value = []
    region_max = int(max(reg.flatten()))  # calculates amount of regions
    for region_number in range(1, region_max + 1):  # iterates over every region
        pos_reg_number = np.where(reg == region_number)
        reg_mean = np.mean(img[pos_reg_number[0], pos_reg_number[1]])
        mean_value.append(reg_mean)
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
    return single_mean


def calculate_one_distance(max_intensity, means, pixel_intensity, region_number):
    """
    calculates the distance of a pixel to the region with a specific region number
    :param max_intensity: maximal intensity value (float)
    :param means: list of means (list)
    :param pixel_intensity: intensity of a pixel (float)
    :param region_number: number of a specific region (int)
    :return: the calculated distance for this pixel (float)
    """
    dist = np.abs((pixel_intensity - means[int(region_number) - 1]) / max_intensity)
    return dist


def calculate_one_border_distances(max_intensity, means, img, max_region, one_border_neighbors):
    """
    calculates the distance of a pixel to the regions of the neighbors of one side
    :param max_intensity: maximal intensity value (float)
    :param means: list of means (list)
    :param img: array with intensity values (2D array)
    :param max_region: highest region number (int)
    :param one_border_neighbors:
    :return: array with the region numbers of the neighbors to one side (2D array)
    """
    one_border_distances = np.ones(img.shape)
    for region_number in range(1, max_region + 1):
        pos_reg_bor = np.where(one_border_neighbors == region_number)
        one_border_distances[pos_reg_bor[0], pos_reg_bor[1]] = calculate_one_distance(max_intensity, means,
                                                                                      img[pos_reg_bor[0],
                                                                                          pos_reg_bor[1]],
                                                                                      region_number)
    return one_border_distances


def calculate_distances(img, reg, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors):
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
    max_intensity = np.amax(img)
    means = mean_region(img, reg)
    max_region = int(np.amax(reg))

    left_distances = calculate_one_border_distances(max_intensity, means, img, max_region, left_neighbors)
    right_distances = calculate_one_border_distances(max_intensity, means, img, max_region, right_neighbors)
    top_distances = calculate_one_border_distances(max_intensity, means, img, max_region, top_neighbors)
    bottom_distances = calculate_one_border_distances(max_intensity, means, img, max_region, bottom_neighbors)

    return means, left_distances, right_distances, top_distances, bottom_distances


def update_list_of_means(means, img, reg, new_pixel):
    """
    updates the list of means with the new mean of the changed region
    :param means: list of means (list)
    :param img: array with intensity values (2D array)
    :param reg: array with region numbers (2D array)
    :param new_pixel: pixel which was lastly assigned to a region (tuple)
    :return: updated list of means (list)
    """
    new_mean = one_region_mean(img, reg, new_pixel)
    means[int(reg[new_pixel] - 1)] = new_mean
    return means


def update_one_distance(img, reg, means, max_intensity, new_pixel, one_border_neighbors, one_border_distances):
    """
    updates the distance array of one direction
    :param img: array with intensity values (2D array)
    :param reg: array with region numbers (2D array)
    :param means: list of means (list)
    :param max_intensity: maximal intensity value (float)
    :param new_pixel: pixel which was lastly assigned to a region (tuple)
    :param one_border_neighbors: array with the region numbers of the neighbors to one side (2D array)
    :param one_border_distances: array with the calculated distances of the pixel to one side (2D array)
    :return: updated array with the calculated distances of the pixel to one side (2D array)
    """
    positions_to_update = np.where(one_border_neighbors == reg[new_pixel])
    one_border_distances[positions_to_update[0],
                         positions_to_update[1]] = calculate_one_distance(max_intensity, means,
                                                                          img[positions_to_update[0],
                                                                              positions_to_update[1]], reg[new_pixel])
    one_border_distances[new_pixel] = 1
    return one_border_distances


def update_distances(img, reg, means, new_pixel, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors,
                     left_distances, right_distances, top_distances, bottom_distances):
    """
    updates the list of means and the four distance arrays for all directions
    :param img: array with intensity values (2D array)
    :param reg: array with region numbers (2D array)
    :param means: list of means (list)
    :param new_pixel: pixel which was lastly assigned to a region (tuple)
    :param left_neighbors: array with the region numbers of the neighbors to the left side (2D array)
    :param right_neighbors: array with the region numbers of the neighbors to the right side (2D array)
    :param top_neighbors: array with the region numbers of the neighbors to the upper side (2D array)
    :param bottom_neighbors: array with the region numbers of the neighbors to lower side (2D array)
    :param left_distances: array with the calculated distances of the pixel to region of left neighbors (2D array)
    :param right_distances: array with the calculated distances of the pixel to region of right neighbors (2D array)
    :param top_distances: array with the calculated distances of the pixel to region of upper neighbors (2D array)
    :param bottom_distances: array with the calculated distances of the pixel to region of lower neighbors (2D array)
    :return: four updated distance arrays (2D arrays)
    """
    means = update_list_of_means(means, img, reg, new_pixel)
    max_intensity = np.amax(img)

    left_distances = update_one_distance(img, reg, means, max_intensity, new_pixel, left_neighbors, left_distances)
    right_distances = update_one_distance(img, reg, means, max_intensity, new_pixel, right_neighbors, right_distances)
    top_distances = update_one_distance(img, reg, means, max_intensity, new_pixel, top_neighbors, top_distances)
    bottom_distances = update_one_distance(img, reg, means, max_intensity, new_pixel, bottom_neighbors,
                                           bottom_distances)

    return left_distances, right_distances, top_distances, bottom_distances


def is_labeled(reg, position):
    """
    determines whether a pixel is already assigned to a region or not
    :param reg: array with region numbers (2D array)
    :param position: pixel that is tested (tuple)
    :return: True/False
    """
    if reg[position] != 0:
        return True
    return False


def unlabeled_pixel_exist(reg):
    """
    determines whether unlabeled pixels exist
    :param reg: array with region numbers (2D array)
    :return: True/False
    """
    if 0 in reg:
        return True
    return False


def position_of_smallest_distance_one_border(one_border_distance):
    """
    finds the position of the pixel with the smallest distance value in an array
    :param one_border_distance: array with distances of the pixels to the neighbors in one direction (2D array)
    :return: pixel with the smallest distance in the array (tuple)
    """
    minimal_distances = np.where(one_border_distance == np.amin(one_border_distance))
    pos_min_dist = list(zip(minimal_distances[0], minimal_distances[1]))[0]
    pos_min_dist = (int(pos_min_dist[0]), int(pos_min_dist[1]))
    return pos_min_dist


def position_of_smallest_distance(left_distances, right_distances, top_distances, bottom_distances):
    """
    finds the pixel with the smallest distance in all four arrays and saves the number of the border
    :param left_distances: array with the calculated distances of the pixel to region of left neighbors (2D array)
    :param right_distances: array with the calculated distances of the pixel to region of right neighbors (2D array)
    :param top_distances: array with the calculated distances of the pixel to region of upper neighbors (2D array)
    :param bottom_distances: array with the calculated distances of the pixel to region of lower neighbors (2D array)
    :return: position of the pixel with the smallest distance (tuple), number ob the used border array (int)
    """
    left_min_dist = position_of_smallest_distance_one_border(left_distances)
    right_min_dist = position_of_smallest_distance_one_border(right_distances)
    top_min_dist = position_of_smallest_distance_one_border(top_distances)
    bottom_min_dist = position_of_smallest_distance_one_border(bottom_distances)

    pos_min_distances = [left_min_dist, right_min_dist, top_min_dist, bottom_min_dist]
    min_distances = np.asarray(
        [left_distances[left_min_dist], right_distances[right_min_dist], top_distances[top_min_dist],
         bottom_distances[bottom_min_dist]])
    border_number = np.where(min_distances == np.amin(min_distances))  # left:0, right:1, top:2, bottom:3
    border_number = border_number[0][0]
    pos_min_dist = pos_min_distances[border_number]

    return pos_min_dist, border_number


def choose_border_number(border_number, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors):
    """
    chooses one of the four neighbor arrays by the number of the array
    :param border_number: number of the array which should be used (int)
    :param left_neighbors: array with the region numbers of the neighbors to the left side (2D array)
    :param right_neighbors: array with the region numbers of the neighbors to the right side (2D array)
    :param top_neighbors: array with the region numbers of the neighbors to the upper side (2D array)
    :param bottom_neighbors: array with the region numbers of the neighbors to lower side (2D array)
    :return: name of the neighbor array which should be used (name of variable)
    """
    if border_number == 0:
        return left_neighbors
    elif border_number == 1:
        return right_neighbors
    elif border_number == 2:
        return top_neighbors
    return bottom_neighbors


def update_left_neighbors(reg, left_neighbors, pos_min_dist):
    """
    updates the left neighbor array
    :param reg: array with region numbers (2D array)
    :param left_neighbors: array with the region numbers of the neighbors to the left side (2D array)
    :param pos_min_dist: pixel which was lastly added to a region (tuple)
    :return: updated left neighbor array (2D array)
    """
    if pos_min_dist[1] != 0:  # not border
        left_neighbor_pos = (pos_min_dist[0], pos_min_dist[1] - 1)
        if not is_labeled(reg, left_neighbor_pos):
            left_neighbors[left_neighbor_pos] = reg[pos_min_dist]
    left_neighbors[pos_min_dist] = 0
    return left_neighbors


def update_right_neighbors(reg, right_neighbors, pos_min_dist):
    """
    updates the right neighbor array
    :param reg: array with region numbers (2D array)
    :param right_neighbors: array with the region numbers of the neighbors to the right side (2D array)
    :param pos_min_dist: pixel which was lastly added to a region (tuple)
    :return: updated right neighbor array (2D array)
    """
    if pos_min_dist[1] != reg.shape[1] - 1:  # not border
        right_neighbor_pos = (pos_min_dist[0], pos_min_dist[1] + 1)
        if not is_labeled(reg, right_neighbor_pos):
            right_neighbors[right_neighbor_pos] = reg[pos_min_dist]
    right_neighbors[pos_min_dist] = 0
    return right_neighbors


def update_top_neighbors(reg, top_neighbors, pos_min_dist):
    """
    updates the upper neighbor array
    :param reg: array with region numbers (2D array)
    :param top_neighbors: array with the region numbers of the neighbors to the upper side (2D array)
    :param pos_min_dist: pixel which was lastly added to a region (tuple)
    :return: updated upper neighbor array (2D array)
    """
    if pos_min_dist[0] != 0:  # not border
        top_neighbor_pos = (pos_min_dist[0] - 1, pos_min_dist[1])
        if not is_labeled(reg, top_neighbor_pos):
            top_neighbors[top_neighbor_pos] = reg[pos_min_dist]
    top_neighbors[pos_min_dist] = 0
    return top_neighbors


def update_bottom_neighbors(reg, bottom_neighbors, pos_min_dist):
    """
    updates the lower neighbor array
    :param reg: array with region numbers (2D array)
    :param bottom_neighbors: array with the region numbers of the neighbors to lower side (2D array)
    :param pos_min_dist: pixel which was lastly added to a region (tuple)
    :return: updated lower neighbor array (2D array)
    """
    if pos_min_dist[0] != reg.shape[0] - 1:  # not border
        bottom_neighbor_pos = (pos_min_dist[0] + 1, pos_min_dist[1])
        if not is_labeled(reg, bottom_neighbor_pos):
            bottom_neighbors[bottom_neighbor_pos] = reg[pos_min_dist]
    bottom_neighbors[pos_min_dist] = 0
    return bottom_neighbors


def label_new_pixel(reg, left_distances, right_distances, top_distances, bottom_distances, left_neighbors,
                    right_neighbors, top_neighbors, bottom_neighbors):
    """
    determines the pixel with the smallest distance and decides how it should be labeled and
    updates the neighbor arrays for all directions
    :param reg: array with region numbers (2D array)
    :param left_distances: array with the calculated distances of the pixel to region of left neighbors (2D array)
    :param right_distances: array with the calculated distances of the pixel to region of right neighbors (2D array)
    :param top_distances: array with the calculated distances of the pixel to region of upper neighbors (2D array)
    :param bottom_distances: array with the calculated distances of the pixel to region of lower neighbors (2D array)
    :param left_neighbors: array with the region numbers of the neighbors to the left side (2D array)
    :param right_neighbors: array with the region numbers of the neighbors to the right side (2D array)
    :param top_neighbors: array with the region numbers of the neighbors to the upper side (2D array)
    :param bottom_neighbors: array with the region numbers of the neighbors to lower side (2D array)
    :return: reg: array with region numbers (2D array), pos_min_dist: pixel that was lastly added to a region (tuple),
    four arrays with region numbers of neighbors (2D arrays)
    """
    pos_min_dist_return = position_of_smallest_distance(left_distances, right_distances, top_distances,
                                                        bottom_distances)
    pos_min_dist = pos_min_dist_return[0]
    border_number = pos_min_dist_return[1]
    one_border_neighbors = choose_border_number(border_number, left_neighbors, right_neighbors, top_neighbors,
                                                bottom_neighbors)

    reg[pos_min_dist] = one_border_neighbors[pos_min_dist]

    left_neighbors = update_left_neighbors(reg, left_neighbors, pos_min_dist)
    right_neighbors = update_right_neighbors(reg, right_neighbors, pos_min_dist)
    top_neighbors = update_top_neighbors(reg, top_neighbors, pos_min_dist)
    bottom_neighbors = update_bottom_neighbors(reg, bottom_neighbors, pos_min_dist)

    return reg, pos_min_dist, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors  # one_border?


def region_growing(img, reg):
    """
    performs region growing algorithm on image with defined seeds (reg)
    :param img: intensity values (2D array)
    :param reg: region numbers (2D array)
    :return: labeled image with region numbers (2D array)
    """
    left_neighbors, right_neighbors, top_neighbors, bottom_neighbors = find_seed_neighbors(reg)

    means, left_distances, right_distances, top_distances, bottom_distances = \
        calculate_distances(img, reg, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors)

    reg, pos_min_dist, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors = \
        label_new_pixel(reg, left_distances, right_distances, top_distances, bottom_distances, left_neighbors,
                        right_neighbors, top_neighbors, bottom_neighbors)

    #  i = 0

    while unlabeled_pixel_exist(reg):
        
        #  i += 1
        #  if i % 5000 == 0:
            #  print(i)

        left_distances, right_distances, top_distances, bottom_distances = \
            update_distances(img, reg, means, pos_min_dist, left_neighbors, right_neighbors, top_neighbors,
                             bottom_neighbors, left_distances, right_distances, top_distances, bottom_distances)

        reg, pos_min_dist, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors = \
            label_new_pixel(reg, left_distances, right_distances, top_distances, bottom_distances, left_neighbors,
                            right_neighbors, top_neighbors, bottom_neighbors)

        # print(np.count_nonzero(reg == 0))
    return reg


if __name__ == '__main__':
    image_intensity = sk.imread("../Data/N2DH-GOWT1/img/t31.tif")  # load image
    # image_intensity = image_intensity[300:350, 450:500]
    # image_r = ip.image_clipping_extreme(image_intensity, 15, 50)
    # image_r = sd.seeds(image_r, 1)
    # image_r = sd.seed_merging(image_r)
    # image_seeds = Image.fromarray(image_r)
    # image_seeds.save("../Result_Pictures/Seeded_Region_Growing/N2DH-GOWT1/t31.tif_seeds.tif")
    image_r = sk.imread("../Result_Pictures/Seeded_Region_Growing/N2DH-GOWT1/")
    image_r = region_growing(image_intensity, image_r)

    im = Image.fromarray(image_r)
    im.save("../Result_Pictures/Seeded_Region_Growing/N2DH-GOWT1/")

