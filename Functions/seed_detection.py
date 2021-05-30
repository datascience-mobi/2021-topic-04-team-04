import skimage.io as sk
import matplotlib.pyplot as plt
import numpy as np
import math as m
from collections import Counter

def standarddeviation(img, size): # calculates standard deviation of every pixel (image, size of filter mask)
    result = np.zeros(img.shape) # c'reate empty array (zeros)
    n = (size - 1) // 2
    for p in np.ndindex(img.shape): # iterates over every pixel
        neighborhood_sum = 0
        mean = 0
        deviation = 0
        if p[0]-n >= 0 and p[1]-n >= 0 and p[0]+n <= img.shape[0] -1 and p[1]+n <= img.shape[1] -1: # no calculation of border pixels
            for q in np.ndindex(size, size): # iterates over filter mask (nxn neighborhood)
                i = p[0]-n + q[0]
                j = p[1]-n + q[1]
                neighborhood_sum += img[i,j]
            mean = neighborhood_sum/(size**2) # calculate mean in nxn neighborhood
            for q in np.ndindex(size, size): # iterates over filter mask (nxn neighborhood)
                i = p[0]-n + q[0]
                j = p[1]-n + q[1]
                deviation += (img[i,j] - mean)**2 # calculate deviation from mean

            result[p] = m.sqrt(1/(size**2) * deviation) # calculate standard deviation
    return result

def euclidean_relative(img, size): # calculates maximum relative euclidean distance for every pixel in nxn neighborhood
    result = np.zeros(img.shape) # create empty array (zeros)
    n = (size - 1) // 2
    for p in np.ndindex(img.shape): # iterates over every pixel using a touple
        neighborhood_distance = [] # creates a list for distances
        if p[0]-n >= 0 and p[1]-n >= 0 and p[0]+n <= img.shape[0] -1 and p[1]+n <= img.shape[1] -1: # no calculation of border pixels
            for q in np.ndindex(size, size): # iterates over nxn neighborhood
                i = p[0]-n + q[0]
                j = p[1]-n + q[1]
                neighborhood_distance.append((img[p]-img[i,j])/img[p]) # adds relative euclidean distance to list

            result[p] = max(neighborhood_distance) # chooses maximum distance
    return result

def euclidean_n(img, size): # calculates maximum euclidean distance for every pixel in nxn neighborhood
    result = np.zeros(img.shape) # create empty array (zeros)
    n = (size - 1) // 2
    for p in np.ndindex(img.shape): # iterates over every pixel using a touple
        neighborhood_distance = [] # creates a list for distances
        if p[0]-n >= 0 and p[1]-n >= 0 and p[0]+n <= img.shape[0] -1 and p[1]+n <= img.shape[1] -1: # no calculation of border pixels
            for q in np.ndindex(size, size): # iterates over nxn neighborhood
                i = p[0]-n + q[0]
                j = p[1]-n + q[1]
                neighborhood_distance.append((img[p]-img[i,j])) # adds euclidean distance to list
            result[p] = max(neighborhood_distance) # chooses maximum distance
    return result

def seeds(img, T1, T2): # automatic seed selection algorithm
    result = np.zeros(img.shape)
    sd_seeds = standarddeviation(img,3) # standard deviation
    sd_flat = sd_seeds.flatten() # standard deviation as 1D-array
    similarity_seeds = 1 - sd_seeds/max(sd_flat) # calculates similarity of every pixel to its neighbors
    eurel_seeds = euclidean_relative(img, 3) # relative euclidean distance of every pixel to its neighbors
    for p in np.ndindex(img.shape): # border pixel value is zero
        if similarity_seeds[p] > T1 and eurel_seeds[p] < T2: # compares pixel with threshold
            result[p] = 1 # assigns value 1 to seeds
    return result

def seed_merging(img):
    regions = np.zeros(img.shape) #creates new array for region numbers for every image pixel
    count = 1 #keep track of region number
    for p in np.ndindex(img.shape): #iterates over every pixel of the image
        if img[p] == 1: #tests if pixel is seed
            if p[0]-1 >= 0 and p[1]-1 >= 0 and p[0]+1 <= img.shape[0] -1 and p[1]+1 <= img.shape[1] -1: # no calculation of border pixels
                for q in np.ndindex(3, 3): # iterates over 3x3 neighborhood
                    i = p[0]-1 + q[0]
                    j = p[1]-1 + q[1]
                    if regions[i,j] != 0: #tests if neighbors are also seeds
                        regions[p] = regions[i,j] # merge neighboring seeds; wird zur letzten abgefragten Region zugeordnet
                if regions[p] == 0: #tests if no neighbors are seeds
                    regions[p] = count #creates new region from new seed
                    count += 1
    return regions

#reduce number of starting regions for region growing by only considering starting regions with more than T seeds
def decrease_region_number(img, T):
    list = Counter(img.flatten()) #counts number of seeds in region
    d_seeds = img.copy()
    for i in range(1, int(np.amax(img))): #iterates over every region
        if list[i] <= T:#if number of seeds is smaller than threshold, delete region
            for p in np.ndindex(img.shape):
                if img[p] == i:
                    d_seeds[p] = 0
    return d_seeds