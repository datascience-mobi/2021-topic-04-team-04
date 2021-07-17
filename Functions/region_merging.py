import numpy as np
from Functions import image_processing as ip
from Functions import seeded_region_growing as srg


def find_neighboring_regions(reg):
    """
    determines adjacent regions of every region in an array where columns and row represent regions
    :param reg: array with region numbers (2D array)
    :return: 2D array where the rows are the regions and the columns are the neighboring regions.
    If a region is a neighbor, True is written in the cell, if not False. Example: 4 is a neighbor of 2: row 2 col 4: 1
    At the beginning, this array is symmetrical (2D array)
    """
    max_region = max(reg.flatten())
    inter_region_neighbors = np.full((int(max_region), int(max_region)), False)
    for region_number in range(1, int(max_region) + 1):
        neighboring_regions = find_neighbors_one_region(reg, region_number)
        inter_region_neighbors[int(region_number) - 1, neighboring_regions - 1] = True
    return inter_region_neighbors


def find_neighbors_one_region(reg, region_number):
    """
    finds all neighbors for a specific region (region_number)
    :param reg: array with region numbers (2D array)
    :param region_number: number of a region (int)
    :return: array with all region numbers of neighbors from region_number (1D array)
    """
    neighboring_regions = np.zeros((reg.shape[0] + 2, reg.shape[1] + 2))
    reg_with_border = ip.add_border(reg)
    reg_with_border = reg_with_border.astype(int)
    pos_pixel_region = np.where(reg_with_border == region_number)
    neighboring_regions[pos_pixel_region[0], pos_pixel_region[1] - 1] = reg_with_border[
        pos_pixel_region[0], pos_pixel_region[1] - 1]
    neighboring_regions[pos_pixel_region[0], pos_pixel_region[1] + 1] = reg_with_border[
        pos_pixel_region[0], pos_pixel_region[1] + 1]
    neighboring_regions[pos_pixel_region[0] - 1, pos_pixel_region[1]] = reg_with_border[
        pos_pixel_region[0] - 1, pos_pixel_region[1]]
    neighboring_regions[pos_pixel_region[0] + 1, pos_pixel_region[1]] = reg_with_border[
        pos_pixel_region[0] + 1, pos_pixel_region[1]]

    neighboring_regions[pos_pixel_region[0], pos_pixel_region[1]] = 0
    neighboring_regions_unique = np.unique(neighboring_regions.flatten())
    pos_zero = np.where(neighboring_regions_unique == 0)
    neighboring_regions_unique = np.delete(neighboring_regions_unique, pos_zero[0])
    neighboring_regions_unique = neighboring_regions_unique.astype(int)
    return neighboring_regions_unique


def region_distance(img, reg):
    """
    calculates distance between all regions
    :param img: intensity values (2D array)
    :param reg: region numbers (2D array)
    :return: 2D array with distances between all regions, only in upper triangle
    (region numbers are row and col number)
    inter_region_distances is upper triangle matrix with mean intensity distances between regions (2D array)
    mean values of all regions (list of floats)
    inter_region_neighbors: all neighbors have 1 in an array where region numbers are row and col numbers (2D array)
    """
    max_region = int(max(reg.flatten()))
    max_intensity = np.amax(img)
    inter_region_distances = np.ones((max_region, max_region))
    means = srg.mean_region(img, reg)
    inter_region_neighbors = find_neighboring_regions(reg)

    for row_number in range(0, int(max_region)):
        neighboring_regions = np.where(inter_region_neighbors[row_number, :] == 1)[0]
        for col_number in neighboring_regions:
            inter_region_distances[row_number][col_number] = distance_between_regions(row_number, col_number,
                                                                                      max_intensity, means)
    inter_region_distances = np.triu(inter_region_distances)
    lower_triangle = np.where(inter_region_distances == 0)
    inter_region_distances[lower_triangle[0], lower_triangle[1]] = 1
    return inter_region_distances, means, inter_region_neighbors


def distance_between_regions(region1, region2, max_intensity, means):
    """
    calculates the distance between two regions
    :param region1: number of region1 -1 (matches the index in the distance, neighbors array, region 1 has index 0, int)
    :param region2: number of region1 -2 (matches the index in the distance, neighbors array, region 1 has index 0, int)
    :param max_intensity: maximal intensity (float)
    :param means: list of means (list of floats)
    :return: distance between the two regions (float)
    """
    distance = abs(means[region1] - means[region2]) / max_intensity
    return distance


def one_merged_region_mean(img, reg, region_number):
    """
    calculates the mean of changed region
    :param img: intensity values (2D array)
    :param reg: region numbers (2D array)
    :param region_number: number of the changed region (int)
    :return: mean value of changed position (float)
    """
    pos_new_reg = np.where(reg == region_number)
    single_mean = np.mean(img[pos_new_reg[0], pos_new_reg[1]])
    return single_mean


def region_distance_new(img, reg, pos_min_dist, means, inter_region_distances, inter_region_neighbors):
    """
    updates array of distances of mean intensity values between all changed regions
    :param img: intensity values (2D array)
    :param reg: region numbers (2D array)
    :param pos_min_dist: tuple with the smallest distance (coordinates are the two regions, both -1 to match the index)
    :param means: mean values of all regions (list of floats)
    :param inter_region_distances: distances between mean intensity values of regions (2D array)
    :param inter_region_neighbors: rows are the regions and the columns are the neighboring regions (2D array)
    :return: inter_region_distances: updated distances between mean intensity values of regions (2D array)
    """
    maximal_intensity = np.amax(img)
    changed_region1 = int(pos_min_dist[0])  # new region is region 1
    changed_region2 = int(pos_min_dist[1])
    means = update_mean_values(means, changed_region1, changed_region2, img, reg)
    inter_region_distances = update_distances(changed_region1, changed_region2, inter_region_distances, means,
                                              maximal_intensity, inter_region_neighbors)
    return inter_region_distances


def update_distances(changed_region1, changed_region2, inter_region_distances, means, maximal_intensity,
                     inter_region_neighbors):
    """
    updates distance values of changed regions, value 1 for removed regions
    :param changed_region1: resulting region number for merged region (-1 to match the index) (int)
    :param changed_region2: region number which is going to be removed (-1 to match the index) (int)
    :param inter_region_distances: distances between mean intensity values of regions (2D array)
    :param means: mean intensity values of regions (list)
    :param maximal_intensity: maximal intensity value of image (float)
    :param inter_region_neighbors: rows are the regions and the columns are the neighboring regions (2D array)
    :return: updated inter_region distances (2D array)
    """
    neighboring_regions = np.where(inter_region_neighbors[changed_region1, :] != 0)[0]
    for element in neighboring_regions:
        if element > changed_region1:
            inter_region_distances[changed_region1, element] = distance_between_regions(changed_region1, element,
                                                                                        maximal_intensity, means)
        elif element < changed_region1:
            inter_region_distances[element, changed_region1] = distance_between_regions(changed_region1, element,
                                                                                        maximal_intensity, means)
    inter_region_distances[changed_region2, :] = 1
    inter_region_distances[:, changed_region2] = 1
    return inter_region_distances


def update_neighboring_regions(inter_region_neighbors, changed_region1, changed_region2):
    """
    updates neighboring regions of changed regions
    :param inter_region_neighbors: all neighbors have values from 1 or higher in an array where region numbers are row
    and col numbers (2D array)
    :param changed_region1: merged region number (-1 to match the index) (int)
    :param changed_region2: region that doesn't exist anymore because of merging (-1 to match the index) (int)
    :return: updated inter_region_neighbors (2D array)
    """
    neighbors_changed_region2 = np.where(inter_region_neighbors[changed_region2, :] == 1)[0]
    inter_region_neighbors[neighbors_changed_region2, changed_region1] = 1
    inter_region_neighbors[changed_region1, neighbors_changed_region2] = 1
    inter_region_neighbors[changed_region2, :] = 0
    inter_region_neighbors[:, changed_region2] = 0
    inter_region_neighbors[changed_region1, changed_region1] = 0
    return inter_region_neighbors


def update_mean_values(means, changed_region1, changed_region2, img, reg):
    """
    updates mean value for merged regions in list of mean values, value 500 for unused means
    :param means: means of intensity for regions (list of floats)
    :param changed_region1: region number for merged region (-1 to match the index) (int)
    :param changed_region2: region number to be removed (-1 to match the index) (int)
    :param img: intensity values (2D array)
    :param reg: region numbers (2D array)
    :return: updated list of means (list of floats)
    """
    means[changed_region2] = 500
    means[int(changed_region1)] = one_merged_region_mean(img, reg, int(changed_region1) + 1)
    return means


def position_of_minimal_distance(inter_region_distances, min_distance):
    """
    searches the two regions with the minimal difference of the intensity value
    :param inter_region_distances: distances between mean intensity values of regions (2D array)
    :param min_distance: minimum of inter_region_distance array (float)
    :return: tuple with the position that shows the regions with the minimal distance mean (both - to match the index)
    """
    minimal_distances = np.where(inter_region_distances == min_distance)
    pos_min_dist = list(zip(minimal_distances[0], minimal_distances[1]))[0]
    pos_min_dist = (int(pos_min_dist[0]), int(pos_min_dist[1]))
    return pos_min_dist


def updates_region_numbers(inter_region_distances, reg, min_distance):
    """
    changes region number to region number of merged region
    :param inter_region_distances: distances between mean intensity values of regions (2D array)
    :param reg: region numbers (2D array)
    :param min_distance: minimal distance of mean intensity values between the regions (float)
    :return: reg: updated region numbers (2D array)
    :return: pos_min_dist: position of minimal distance in inter_region_distances array (tuple)
    """
    pos_min_dist = position_of_minimal_distance(inter_region_distances, min_distance)
    pixel_to_change = np.where(reg == pos_min_dist[1] + 1)
    #  column in inter_region_distances is region to be changed(column number bigger than row number)
    pixel_to_change_rows = pixel_to_change[0]
    pixel_to_change_cols = pixel_to_change[1]
    reg[pixel_to_change_rows, pixel_to_change_cols] = pos_min_dist[0] + 1
    return reg, pos_min_dist


def distance_merging_while(reg, threshold, img):
    """
    region merging algorithm by similarity of mean intensity values of regions
    :param reg: region numbers (2D array)
    :param threshold: distance intensity value below which regions are merged (float between 0 and 1)
    :param img: intensity value (2D array)
    :return: merged regions by intensity similarity (2D array)
    """
    inter_region_distances, means, inter_region_neighbors = region_distance(img, reg)
    min_distance = np.nanmin(inter_region_distances)
    #  print(np.amax(reg))
    #  i = 1
    while minimal_distance_is_similar(threshold, min_distance):
        #  i += 1
        #  if i % 100 == 0:
            #  print(i)

        reg, pos_min_dist = updates_region_numbers(inter_region_distances, reg, min_distance)
        inter_region_neighbors = update_neighboring_regions(inter_region_neighbors, pos_min_dist[0], pos_min_dist[1])
        inter_region_distances = region_distance_new(img, reg, pos_min_dist, means, inter_region_distances,
                                                     inter_region_neighbors)
        min_distance = np.nanmin(inter_region_distances)
    return reg, inter_region_neighbors, means


def minimal_distance_is_similar(threshold, min_distance):
    """
    compares minimal distance between 2 regions to a threshold
    :param threshold: distance intensity value below which regions are merged (float between 0 and 1)
    :param min_distance: minimal distance of mean intensity values between the regions (float)
    :return: True when minimal distance is below the threshold and False if not
    """
    if min_distance < threshold:
        return True
    return False


def calculate_regions_size(regions):
    """
    calculates number of assigned pixels of every region
    :param regions: array with region numbers (2D array)
    :return: array with region size of every region (1D array with ints) (index is -1: region 1 has index 0)
    if region is empty because it is already merged to another one the value nan is assigned (no merging)
    """
    max_region = np.amax(regions)
    region_sizes = []
    for region_number in range(1, int(max_region) + 1):
        region_count = np.sum(regions == region_number)
        region_sizes.append(region_count)
    region_sizes = np.asarray(region_sizes, float)
    pos_empty_regions = np.where(region_sizes == 0)[0]
    region_sizes[pos_empty_regions] = np.nan
    return region_sizes


def find_smallest_region(region_sizes):
    """
    determines smallest region
    :param region_sizes: array of size of every region (1D array with ints)
    :return: region number of the smallest region (int) (-1 to match index)
    """
    smallest_size = np.nanmin(region_sizes)
    smallest_region = np.where(region_sizes == smallest_size)[0]
    smallest_region = smallest_region[0]
    return smallest_region


def find_most_similar_region(means, smallest_region, inter_region_neighbors, img):
    """
    determines the region with the most similar mean intensity value of the smallest region
    :param means: mean values of all regions (list of floats)
    :param smallest_region: region number of the smallest region (int) (-1 to match index)
    :param inter_region_neighbors: all neighbors have values from 1 or higher in an array where region numbers are row
    and col numbers (2D array)
    :param img: array with intensity values (2D array)
    :return: region number with the most similar mean intensity value of the smallest region (int) (-1 to match index)
    """
    means = np.asarray(means)
    max_intensity = int(np.amax(img))
    distances = np.ones(means.shape)
    neighboring_regions = np.where(inter_region_neighbors[smallest_region, :] != 0)[0]
    distances[neighboring_regions] = abs(means[neighboring_regions] - means[smallest_region]) / max_intensity
    smallest_distance = np.amin(distances)
    closest_neighbor = np.where(distances == smallest_distance)[0]
    closest_neighbor = closest_neighbor[0]
    return closest_neighbor  # number of region starts with 0


def update_regions(reg, closest_neighbor, smallest_region):  # merging
    """
    updates region numbers
    :param reg: array with region numbers (2D array)
    :param closest_neighbor: region number of the region with the most similar mean intensity to smallest region (int)
    (-1 to match index)
    :param smallest_region: region number of the smallest region (int) (-1 to match index)
    :return: updated region number array (2D array)
    """
    pos_smallest_region = np.where(reg == smallest_region + 1)
    reg[pos_smallest_region[0], pos_smallest_region[1]] = closest_neighbor + 1
    return reg


def update_region_sizes(region_sizes, smallest_region, closest_neighbor):
    """
    updates region_sizes
    :param region_sizes: size of every region (1D array with ints)
    :param smallest_region: region number of the smallest region (int) (-1 to match index)
    :param closest_neighbor: region number of the region with the most similar mean intensity to smallest region (int)
    (-1 to match index)
    :return: updates array with region size of every region (1D array with ints)
    """
    region_sizes[closest_neighbor] = region_sizes[smallest_region] + region_sizes[closest_neighbor]
    region_sizes[smallest_region] = np.nan
    return region_sizes


def region_merging_size(img, reg, inter_region_neighbors, means, threshold):
    """
    region merging algorithm by size of regions
    :param img: array with intensity values (2D array)
    :param reg: array with region numbers (2D array)
    :param inter_region_neighbors: all neighbors have values from 1 or higher in an array where region numbers are row
    and col numbers (2D array)
    :param means: mean values of all regions (list of floats)
    :param threshold: size value below which regions are merged (int)
    :return: merged regions by size (2D array)
    """
    region_sizes = calculate_regions_size(reg)
    smallest_region = find_smallest_region(region_sizes)
    #  print(len(np.unique(reg.flatten())))
    #  i = 0
    while region_sizes[smallest_region] < threshold:
        #  i += 1
        #  if i % 100 == 0:
            #  print(i)

        closest_neighbor = find_most_similar_region(means, smallest_region, inter_region_neighbors, img)
        reg = update_regions(reg, closest_neighbor, smallest_region)
        means = update_mean_values(means, closest_neighbor, smallest_region, img, reg)
        region_sizes = update_region_sizes(region_sizes, smallest_region, closest_neighbor)
        inter_region_neighbors = update_neighboring_regions(inter_region_neighbors, closest_neighbor, smallest_region)
        smallest_region = find_smallest_region(region_sizes)
    return reg


def region_merging(reg, img, distance_threshold, size_threshold):
    """
    performs the distance region merging algorithm and the size region merging algorithm after each other
    :param reg: array with region numbers (2D array)
    :param img: array with intensity values (2D array)
    :param distance_threshold: regions with smaller intensity distance than the threshold will be merged (float)
    :param size_threshold: regions that are smaller than this threshold will be merged (int)
    :return:
    """
    results_region_merging_similarity = distance_merging_while(reg, distance_threshold, img)
    image_rm_similarity, inter_region_neighbors, means = results_region_merging_similarity
    image_rm_size = region_merging_size(img, image_rm_similarity, inter_region_neighbors, means, size_threshold)
    return image_rm_size
