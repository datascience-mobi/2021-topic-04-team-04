import numpy as np
from Functions import seeded_region_growing as srg


#  from Functions import image_processing as ip
#  from PIL import Image
#  import skimage.io as sk


def unseeded_calculate_one_distance(means, pixel_intensity, region_number):
    dist = np.abs(pixel_intensity - means[int(region_number) - 1])
    return dist


def unseeded_calculate_one_border_distances(means, img, max_region, one_border_neighbors):
    one_border_distances = np.full(img.shape, 500)
    for region_number in range(1, max_region + 1):
        pos_reg_bor = np.where(one_border_neighbors == region_number)
        one_border_distances[pos_reg_bor[0], pos_reg_bor[1]] = \
            unseeded_calculate_one_distance(means, img[pos_reg_bor[0], pos_reg_bor[1]], region_number)
    return one_border_distances


def unseeded_calculate_distances(img, reg, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors):
    means = srg.mean_region(img, reg)
    max_region = int(np.amax(reg))

    left_distances = unseeded_calculate_one_border_distances(means, img, max_region, left_neighbors)
    right_distances = unseeded_calculate_one_border_distances(means, img, max_region, right_neighbors)
    top_distances = unseeded_calculate_one_border_distances(means, img, max_region, top_neighbors)
    bottom_distances = unseeded_calculate_one_border_distances(means, img, max_region, bottom_neighbors)

    return means, left_distances, right_distances, top_distances, bottom_distances


def unseeded_update_one_distance(img, reg, means, new_pixel, one_border_neighbors, one_border_distances):
    positions_to_update = np.where(one_border_neighbors == reg[new_pixel])
    one_border_distances[positions_to_update[0],
                         positions_to_update[1]] = unseeded_calculate_one_distance(means, img[positions_to_update[0],
                                                                                              positions_to_update[1]],
                                                                                   reg[new_pixel])
    one_border_distances[new_pixel] = 500
    return one_border_distances


def unseeded_update_distances(img, reg, means, new_pixel, left_neighbors, right_neighbors, top_neighbors,
                              bottom_neighbors,
                              left_distances, right_distances, top_distances, bottom_distances):
    means = srg.update_list_of_means(means, img, reg, new_pixel)

    left_distances = unseeded_update_one_distance(img, reg, means, new_pixel, left_neighbors, left_distances)
    right_distances = unseeded_update_one_distance(img, reg, means, new_pixel, right_neighbors, right_distances)
    top_distances = unseeded_update_one_distance(img, reg, means, new_pixel, top_neighbors, top_distances)
    bottom_distances = unseeded_update_one_distance(img, reg, means, new_pixel, bottom_neighbors,
                                                    bottom_distances)

    return left_distances, right_distances, top_distances, bottom_distances, means


def unseeded_choose_distance_array(border_number, left_distances, right_distances, top_distances, bottom_distances):
    if border_number == 0:
        return left_distances
    elif border_number == 1:
        return right_distances
    elif border_number == 2:
        return top_distances
    return bottom_distances


# unseeded_label new pixel

def unseeded_label_new_pixel(reg, left_distances, right_distances, top_distances, bottom_distances, left_neighbors,
                             right_neighbors,
                             top_neighbors, bottom_neighbors, t, means, img):
    pos_min_dist, border_number = srg.position_of_smallest_distance(left_distances, right_distances, top_distances,
                                                                    bottom_distances)
    one_border_neighbors = srg.choose_border_number(border_number, left_neighbors, right_neighbors, top_neighbors,
                                                    bottom_neighbors)
    one_border_distances = unseeded_choose_distance_array(border_number, left_distances, right_distances, top_distances,
                                                          bottom_distances)

    if one_border_distances[pos_min_dist] < t:
        reg[pos_min_dist] = one_border_neighbors[pos_min_dist]
    else:
        distance_list = []
        for m in means:
            distance_list.append(np.abs(img[pos_min_dist] - m))
        minimum = min(distance_list)
        if minimum < t:
            reg[pos_min_dist] = distance_list.index(minimum) + 1
        else:
            region_max = int(max(reg.flatten()))
            reg[pos_min_dist] = region_max + 1
            means.append(img[pos_min_dist])

    left_neighbors = srg.update_left_neighbors(reg, left_neighbors, pos_min_dist)
    right_neighbors = srg.update_right_neighbors(reg, right_neighbors, pos_min_dist)
    top_neighbors = srg.update_top_neighbors(reg, top_neighbors, pos_min_dist)
    bottom_neighbors = srg.update_bottom_neighbors(reg, bottom_neighbors, pos_min_dist)

    return reg, pos_min_dist, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors, means


def unseeded_region_growing_algorithm(img, reg, t):
    left_neighbors, right_neighbors, top_neighbors, bottom_neighbors = srg.find_seed_neighbors(reg)

    means, left_distances, right_distances, top_distances, bottom_distances = \
        unseeded_calculate_distances(img, reg, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors)

    reg, pos_min_dist, left_neighbors, right_neighbors, top_neighbors, bottom_neighbors, means = \
        unseeded_label_new_pixel(reg, left_distances, right_distances, top_distances, bottom_distances, left_neighbors,
                                 right_neighbors, top_neighbors, bottom_neighbors, t, means, img)

    i = 0

    while srg.unlabeled_pixel_exist(reg):

        i += 1
        if i % 1000 == 0:
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
