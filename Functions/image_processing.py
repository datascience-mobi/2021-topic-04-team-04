import skimage.io as sk
import matplotlib.pyplot as plt
import numpy as np
import math as m

def show_image(img, x, y): # Funktion um Bilder schneller schön anzuzeigen
    plt.figure(figsize=(x,y))
    plt.imshow(img, "gray")
    plt.colorbar()

def img_resize(img, size_l, size_b):
    new_size = [size_l, size_b]
    img_new = np.zeros(new_size)
    for x in np.ndindex(new_size[0], new_size[1]):
        img_new[x] = img[x]
    return img_new

def t_pic():
    img = np.zeros([100, 100])
    img[30:40, 30:40] = 1
    img[75:80, 75:80] = 1
    return img