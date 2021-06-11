import skimage.io as sk
import numpy as np
from Functions import image_processing as ip
from Functions import seed_detection as sd


# import math as m
# import matplotlib.pyplot as plt

def find_neighbors(reg):
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


def add_neighbors(img, p):  # p describes pixel for which neighbors need to be added
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
    mean_value = []
    region_max = int(max(reg.flatten()))  # calculates amount of regions
    for count in range(1, region_max + 1):  # iterates over every region
        intensity = []
        for p in np.ndindex(img.shape):  # iterates over every pixel in the image
            if reg[p] == count:
                intensity.append(img[p])  # appends intensity value, if it is in the region
        mean_value.append(np.mean(intensity))  # calculates mean value of region
    return mean_value  # returns list with average of every region


# calculated mean value of region of newly labeled pixel
def one_region_mean(img, regions,
                    new_pixel):  # img is array of intensity values, regions is array with region numbers, new_pixel is position of last added pixel
    intensity = []
    for p in np.ndindex(img.shape):
        if regions[p] == regions[new_pixel]:  # finds region of newly added pixel
            intensity.append(
                img[p])  # iterates over every pixel in the image and appends intensity value, if it is in the region
    single_mean = sum(intensity) / len(intensity)  # calculates mean value of region with new pixel
    return single_mean  # returns mean value of changed region


def calculation_distance(img, Ne, regions):  # img intensity values, regions is region number, Ne is list of neighbours
    means = mean_region(img, regions)  # list of mean values of every region
    result = np.ones(img.shape)  # new array with distance values, standard value is one
    nearest_reg = np.zeros(img.shape)
    for i in Ne:
        nei = add_neighbors(img, i)  # list 4 neighbors of pixel i out of unsorted neighbors list
        distance = []
        for j in nei:
            if regions[j] != 0:  # only neighboring pixels which are sorted
                distance.append(np.abs((img[j] - means[regions[j] - 1])) / img[
                    j])  # saves tuple of normalized distance of pixel to neighbor j and region of j
        min_dist = np.amin(distance)  # saves minimal distance to 1 of its neighbors in distance array
        nearest_reg[i] = regions[np.where(distance == min_dist)]  # saves number of nearest region
        result[i] = min_dist
    return result, nearest_reg  # returns array with distance values between 0 and 1 and array with number of nearest region


# updates distances for updated region
def new_distance(img, regions, nearest_reg, distances, new_pixel, Ne):
    new_mean = one_region_mean(img, regions, new_pixel)
    means = mean_region(img, regions)
    means[regions[new_pixel] - 1] = new_mean  # list of all mean values of the region with the updated region
    min_dist = 1
    for i in Ne:
        nei = add_neighbors(img, i)  # list 4 neighbors of pixel i out of unsorted neighbors list
        distance = []
        for j in nei:
            if regions[nei] == regions[new_pixel]:  # calculates distance only for pixel adjacent to updated region
                for j in nei:
                    if regions[j] != 0:  # compare all neighboring regions
                        distance.append(np.abs((img[j] - means[regions[j] - 1])) / img[
                            j])  # saves tuple of normalized distance of pixel to neighbor j and region of j
                min_dist = np.amin(distance)  # saves minimal distance to 1 of its neighbors in distance array
                continue  # only calculates new distances once
        nearest_reg[i] = regions[np.where(distance == min_dist)]  # saves number of nearest region
        distances[i] = min_dist
    return distances, nearest_reg  # returns array with distance values between 0 and 1 and array with number of nearest region


def label(regions, distances,
          nearest_reg):  # regions is array of region numbers, distances is array of distances, nearest_reg is array of nearest region number
    x = np.where(distances == np.amin(distances))  # finds minimal distance in array and its position
    pos_min_dist = list(zip(x[0], x[1]))[0]  # position of first pixel with minimal distance value
    regions[pos_min_dist] = nearest_reg[pos_min_dist]
    return regions, pos_min_dist  # returns new labeled region array and position of newly labeled pixel


def region_growing(img, regions):
    Ne = find_neighbors(regions)  # list of all adjacent pixels
    dist = calculation_distance(img, Ne, regions)  # array of smallest distances
    regions_new = label(regions, dist[0], dist[1])  # labels pixel with smallest distance
    Ne_add = add_neighbors(img, regions_new[1])  # finds neighbors of newly labeled pixel
    for j in Ne_add:  # updates list of adjacent pixels
        if j not in Ne:
            Ne.append(j)
    while len(Ne) > 0:  # as long as not all pixels are sorted
        dist = new_distance(img, regions_new[0], dist[1], dist[0], regions_new[1], Ne)  # updates distances
        regions_new = label(regions_new[0], dist[0], dist[1])  # labels next pixel
        Ne_add = add_neighbors(img, regions_new[1])  # finds neighbors of newly labeled pixel
        for j in Ne_add:  # updates Ne
            if j not in Ne:
                Ne.append(j)
    return regions_new[0]  # returns array with region numbers


if __name__ == '__main__':
    image = sk.imread("Data/N2DH-GOWT1/img/t01.tif")  # Bild laden
    img_s = image[300:400, 300:500]
    img_result = sd.seeds(img_s, 0.4, 40)
    img_result = sd.seed_merging(img_result)
    img_result = sd.decrease_region_number(img_result, 50)

    img_result = region_growing(img_s, img_result)
    ip.show_image(img_result, 15, 8)
