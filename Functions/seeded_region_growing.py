import skimage.io as sk
import numpy as np
from Functions import image_processing as ip
from Functions import seed_detection as sd
from PIL import Image


def calculate_left_neighbors(reg, seeds):
    left_neighbors = np.zeros((reg.shape[0], reg.shape[1] + 1))
    left_neighbors[seeds[0], seeds[1]] = reg[seeds[0], seeds[1]]
    left_neighbors = np.delete(left_neighbors, 0, axis=1)
    left_neighbors[seeds[0], seeds[1]] = 0  # remove already labeled pixels
    return left_neighbors


def calculate_right_neighbors(reg, seeds):
    right_neighbors = np.zeros((reg.shape[0], reg.shape[1] + 1))
    right_neighbors[seeds[0], seeds[1] + 1] = reg[seeds[0], seeds[1]]
    right_neighbors = np.delete(right_neighbors, reg.shape[1] - 1, axis=1)
    right_neighbors[seeds[0], seeds[1]] = 0
    return right_neighbors


def calculate_top_neighbors(reg, seeds):
    top_neighbors = np.zeros((reg.shape[0] + 1, reg.shape[1]))
    top_neighbors[seeds[0], seeds[1]] = reg[seeds[0], seeds[1]]
    top_neighbors = np.delete(top_neighbors, 0, axis=0)
    top_neighbors[seeds[0], seeds[1]] = 0
    return top_neighbors


def calculate_bottom_neighbors(reg, seeds):
    bottom_neighbors = np.zeros((reg.shape[0] + 1, reg.shape[1]))
    bottom_neighbors[seeds[0] + 1, seeds[1]] = reg[seeds[0], seeds[1]]
    bottom_neighbors = np.delete(bottom_neighbors, reg.shape[0] - 1, axis=0)
    bottom_neighbors[seeds[0], seeds[1]] = 0
    return bottom_neighbors


def find_seed_neighbors(reg):
    seeds = np.where(reg != 0)

    left_neighbors = calculate_left_neighbors(reg, seeds).astype(int)
    right_neighbors = calculate_right_neighbors(reg, seeds).astype(int)
    top_neighbors = calculate_top_neighbors(reg, seeds).astype(int)
    bottom_neighbors = calculate_bottom_neighbors(reg, seeds).astype(int)

    return left_neighbors, right_neighbors, top_neighbors, bottom_neighbors


def mean_region(img,reg):
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
    return single_mean  # returns mean value of changed regio


def calculate_one_distance(max_intensity, means, pixel_intensity, region_number):
    dist = np.abs((pixel_intensity - means[int(region_number) - 1]) / max_intensity)
    return dist


def calculate_one_border_distances(max_intensity, means, img, max_region, one_border_neighbors):
    one_border_distances = np.ones(img.shape)
    for region_number in range(1, max_region + 1):
        pos_reg_bor = np.where(one_border_neighbors == region_number)
        one_border_distances[pos_reg_bor[0], pos_reg_bor[1]] = calculate_one_distance(max_intensity, means,
                                                                                      img[pos_reg_bor[0],
                                                                                          pos_reg_bor[1]],
                                                                                      region_number)
    return one_border_distances


def calculate_distances(img, reg, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors):
    max_intensity = np.amax(img)
    means = mean_region(img, reg)
    max_region = int(np.amax(reg))

    left_distances = calculate_one_border_distances(max_intensity, means, img, max_region, left_neighbors)
    right_distances = calculate_one_border_distances(max_intensity, means, img, max_region, right_neighbors)
    top_distances = calculate_one_border_distances(max_intensity, means, img, max_region, top_neighbors)
    bottom_distances = calculate_one_border_distances(max_intensity, means, img, max_region, bottom_neighbors)

    return means, left_distances, right_distances, top_distances, bottom_distances


def update_list_of_means(means, img, reg, new_pixel):
    new_mean = one_region_mean(img, reg, new_pixel)
    means[int(reg[new_pixel] - 1)] = new_mean
    return means


def update_one_distance(img, reg, means, max_intensity, new_pixel, one_border_neighbors, one_border_distances):
    positions_to_update = np.where(one_border_neighbors == reg[new_pixel])
    one_border_distances[positions_to_update[0],
                         positions_to_update[1]] = calculate_one_distance(max_intensity, means,
                                                                          img[positions_to_update[0],
                                                                              positions_to_update[1]], reg[new_pixel])
    one_border_distances[new_pixel] = 1
    return one_border_distances


def update_distances(img, reg, means, new_pixel, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors,
                     left_distances, right_distances, top_distances, bottom_distances):
    means = update_list_of_means(means, img, reg, new_pixel)
    max_intensity = np.amax(img)

    left_distances = update_one_distance(img, reg, means, max_intensity, new_pixel, left_neighbors, left_distances)
    right_distances = update_one_distance(img, reg, means, max_intensity, new_pixel, right_neighbors, right_distances)
    top_distances = update_one_distance(img, reg, means, max_intensity, new_pixel, top_neighbors, top_distances)
    bottom_distances = update_one_distance(img, reg, means, max_intensity, new_pixel, bottom_neighbors,
                                           bottom_distances)

    return left_distances, right_distances, top_distances, bottom_distances


def is_labeled(reg, position):
    if reg[position] != 0:
        return True
    return False


def unlabeled_pixel_exist(reg):
    if 0 in reg:
        return True
    return False


def position_of_smallest_distance_one_border(one_border_distance):
    minimal_distances = np.where(one_border_distance == np.amin(one_border_distance))
    pos_min_dist = list(zip(minimal_distances[0], minimal_distances[1]))[0]
    pos_min_dist = (int(pos_min_dist[0]), int(pos_min_dist[1]))
    return pos_min_dist


def position_of_smallest_distance(left_distances, right_distances, top_distances, bottom_distances):
    left_min_dist = position_of_smallest_distance_one_border(left_distances)
    right_min_dist = position_of_smallest_distance_one_border(right_distances)
    top_min_dist = position_of_smallest_distance_one_border(right_distances)
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
    if border_number == 0:
        return left_neighbors
    elif border_number == 1:
        return right_neighbors
    elif border_number == 2:
        return top_neighbors
    return bottom_neighbors


def update_left_neighbors(reg, left_neighbors, pos_min_dist):
    if pos_min_dist[1] != 0:  # not border
        left_neighbor_pos = (pos_min_dist[0], pos_min_dist[1] - 1)
        if not is_labeled(reg, left_neighbor_pos):
            left_neighbors[left_neighbor_pos] = reg[pos_min_dist]
    left_neighbors[pos_min_dist] = 0
    return left_neighbors


def update_right_neighbors(reg, right_neighbors, pos_min_dist):
    if pos_min_dist[1] != reg.shape[1] - 1:  # not border
        right_neighbor_pos = (pos_min_dist[0], pos_min_dist[1] + 1)
        if not is_labeled(reg, right_neighbor_pos):
            right_neighbors[right_neighbor_pos] = reg[pos_min_dist]
    right_neighbors[pos_min_dist] = 0
    return right_neighbors


def update_top_neighbors(reg, top_neighbors, pos_min_dist):
    if pos_min_dist[0] != 0:  # not border
        top_neighbor_pos = (pos_min_dist[0] - 1, pos_min_dist[1])
        if not is_labeled(reg, top_neighbor_pos):
            top_neighbors[top_neighbor_pos] = reg[pos_min_dist]
    top_neighbors[pos_min_dist] = 0
    return top_neighbors


def update_bottom_neighbors(reg, bottom_neighbors, pos_min_dist):
    if pos_min_dist[0] != reg.shape[0] - 1:  # not border
        bottom_neighbor_pos = (pos_min_dist[0] + 1, pos_min_dist[1])
        if not is_labeled(reg, bottom_neighbor_pos):
            bottom_neighbors[bottom_neighbor_pos] = reg[pos_min_dist]
    bottom_neighbors[pos_min_dist] = 0
    return bottom_neighbors


def label_new_pixel(reg, left_distances, right_distances, top_distances, bottom_distances, left_neighbors,
                    right_neighbors,
                    top_neighbors, bottom_neighbors):
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
    :param img: intensity value (2d array)
    :param reg: region numbers (2d array)
    :return: labeled image with region numbers (2d array)
    """

    border_neighbors = find_seed_neighbors(reg)
    left_neighbors = border_neighbors[0]
    right_neighbors = border_neighbors[1]
    top_neighbors = border_neighbors[2]
    bottom_neighbors = border_neighbors[3]

    border_distances = calculate_distances(img, reg, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors)
    means = border_distances[0]
    left_distances = border_distances[1]
    right_distances = border_distances[2]
    top_distances = border_distances[3]
    bottom_distances = border_distances[4]

    regions_new = label_new_pixel(reg, left_distances, right_distances, top_distances, bottom_distances,
                                  left_neighbors, right_neighbors, top_neighbors, bottom_neighbors)
    reg = regions_new[0]
    pos_min_dist = regions_new[1]
    left_neighbors = regions_new[2]
    right_neighbors = regions_new[3]
    top_neighbors = regions_new[4]
    bottom_neighbors = regions_new[5]

    while unlabeled_pixel_exist(reg):
        new_distances = update_distances(img, reg, means, pos_min_dist, left_neighbors, right_neighbors, top_neighbors,
                                         bottom_neighbors, left_distances, right_distances, top_distances,
                                         bottom_distances)
        left_distances = new_distances[0]
        right_distances = new_distances[1]
        top_distances = new_distances[2]
        bottom_distances = new_distances[3]

        regions_new = label_new_pixel(reg, left_distances, right_distances, top_distances, bottom_distances,
                                      left_neighbors, right_neighbors, top_neighbors, bottom_neighbors)
        reg = regions_new[0]
        pos_min_dist = regions_new[1]
        left_neighbors = regions_new[2]
        right_neighbors = regions_new[3]
        top_neighbors = regions_new[4]
        bottom_neighbors = regions_new[5]

        # print(np.count_nonzero(reg == 0))
    return reg


if __name__ == '__main__':
    image = sk.imread("../Data/N2DH-GOWT1/img/t01.tif")  # load image
    img_s = image[300:400, 300:500]
    img_result = sd.seeds(img_s, 0.4, 40)
    img_result = sd.seed_merging(img_result)
    img_result = sd.decrease_region_number(img_result, 50)

    img_result = region_growing(img_s, img_result)
    ip.show_image(img_result, 15, 8)

    im = Image.fromarray(img_result)
    im.save("../Result_Pictures/Seeded_Region_Growing/N2DH-GOWT1/srg_t01.tif")
