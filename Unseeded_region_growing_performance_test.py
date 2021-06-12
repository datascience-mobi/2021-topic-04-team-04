from Functions import image_processing as ip
import skimage.io as sk
import numpy as np
from Functions import seeded_region_growing as srg
from Functions import unseeded_region_growing as urg

image = sk.imread("Data/N2DH-GOWT1/img/t01.tif")
img_r = ip.img_resize(image, 500, 500)
ip.show_image(img_r, 15, 8)

img_small = image[300:350, 400:450]
ip.show_image(img_small, 15, 8)

regions = np.zeros(img_small.shape, int)  # array with region number
regions[0, 0] = 1  # Startpixel is region number 1


# eventuell noch Methode um nur neue Distanzen zu berechnen
def unseeded_region_growing_algorithm(img, reg, t):
    neighbors = srg.find_neighbors(reg)
    while len(neighbors) > 0:
        distance_result = urg.unseeded_distance(img, neighbors, reg)
        dis = distance_result[0]
        nearest_region = distance_result[1]
        pick = urg.unseeded_pixel_pick(dis)
        if dis[pick] < t:
            reg = urg.unseeded_region_direct(reg, pick, nearest_region)  # Fehlermeldung nicht verstanden
        else:
            reg = urg.unseeded_region_indirect_or_new(img, reg, pick, t)
        neighbors_add = srg.add_neighbors(img, pick)
        for j in neighbors_add:
            if j not in neighbors and reg[j] == 0:
                neighbors.append(j)
        neighbors.remove(pick)
        dis[pick] = 500
        print(len(neighbors))
    return reg


#  Je kleiner Threshold, desto mehr Regionen und desto länger dauert es
test2 = unseeded_region_growing_algorithm(img_small, regions, 5)
print(test2)
ip.show_image(test2, 15, 8)
