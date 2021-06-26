import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from medpy.filter.smoothing import anisotropic_diffusion


def show_image(img, x, y):
    """ # Funktion um Bilder schneller schön anzuzeigen
    :param img: Bild welches angezeigt werden soll
    :param x: Länge des Bildes in x-Richtung
    :param y: Länge des Bildes in y-Richtung
    :return: 
    """
    plt.figure(figsize=(x, y))
    plt.imshow(img, "gray")
    plt.colorbar()


def img_resize(img, size_x, size_y):
    """ # Funktion erstellt einen Bildausschnitt (oben rechts)
    :param img: Bild aus dem ausgeschnitten wird
    :param size_x: Länge des Bildes in x-Richtung
    :param size_y: Länge des Bildes in y-Richtung
    :return: Erstellter Bildausschnitt
    """
    new_size = [size_y, size_x]
    img_new = np.zeros(new_size)
    for x in np.ndindex(new_size[0], new_size[1]):
        img_new[x] = img[x]
    return img_new


def t_pic():
    """ Funktion erstellt ein schwarzes Bild mit bestimmte weißen Bereichen
    :return: Schwarzes Bild mit bestimmten weißen Bereichen
    """
    img = np.zeros([100, 100])
    img[30:40, 30:40] = 1
    img[75:80, 75:80] = 1
    return img


def is_border_pixel(pixel, img):
    """ Funktion prüft für einen Pixel, ob es sich um einen Randpixel handelt
    :param pixel: Pixel der geprüft wird
    :param img: Bild für welches der Rand geprüft wird
    :return: True für einen Randpixel, False für keinen Randpixel
    """
    if pixel[0] - 1 >= 0 and pixel[1] - 1 >= 0 and pixel[0] + 1 <= img.shape[0] - 1 and pixel[1] + 1 <= img.shape[
        1] - 1:
        return False
    return True


def filter_iteration_sum(img, pixel, size):
    """ Funktion berechnet die Summe der Intensitäten aller Nachbarn in einer Filtermaske
    :param img: Bild dessen Intensitäten verwendet werden
    :param pixel: Pixel auf den die Filtermaske angewendet wird
    :param size: Größe der Filtermaske
    :return: Summer der Intensitäten aller Pixel in der Filtermaske
    """
    n = (size - 1) // 2
    neighborhood_sum = 0
    for filter_pixel in np.ndindex(size, size):
        filter_neighbors_row = pixel[0] + filter_pixel[0] - n
        filter_neighbors_col = pixel[1] + filter_pixel[1] - n
        neighborhood_sum += img[filter_neighbors_row, filter_neighbors_col]
    return neighborhood_sum


def filter_iteration_deviation(img, pixel, size, mean):
    """ # Funktion berechnet die Summe der Abweichungen der Intensitäten vom Mittelwerte der Pixel einer Filtermaske
    :param img: Bild dessen Intensitäten verwendet werden
    :param pixel: Pixel auf den die Filtermakse angewendet wird
    :param size: Größe der Filtermaske
    :param mean: Mittelwert von dem die Abweichung berechnet wird
    :return: Summer der Abweichungen vom Mittelwert
    """
    n = (size - 1) // 2
    deviation = 0
    for filter_pixel in np.ndindex(size, size):
        filter_neighbors_row = pixel[0] + filter_pixel[0] - n
        filter_neighbors_col = pixel[1] + filter_pixel[1] - n
        deviation += (img[filter_neighbors_row, filter_neighbors_col] - mean) ** 2
    return deviation


def add_border(array):
    array_with_border = np.zeros((array.shape[0] + 2, array.shape[1] + 2))
    array_with_border[1:array.shape[0] + 1, 1:array.shape[1] + 1] = array
    return array_with_border


def remove_border(array):
    array = np.delete(array, 0, axis=1)
    array = np.delete(array, array.shape[1] - 1, axis=1)
    array = np.delete(array, 0, axis=0)
    array = np.delete(array, array.shape[0] - 1, axis=0)
    return array


def histogram(img):
    histo = plt.hist(img.flatten(), log=True)
    plt.show(histo)


def image_clipping(img, t1, t2):
    img_copy = img.copy()
    clipped_img = np.clip(img_copy, t1, t2)
    return clipped_img


def image_clipping_extreme(img, t1, t2):
    img_copy = img.copy()
    maxi = max(img.flatten())
    for p in np.ndindex(img_copy.shape):
        if img_copy[p] < t1:
            img_copy[p] = 0
        if img_copy[p] > t2:
            img_copy[p] = maxi
        else:
            img_copy[p] = img_copy[p]

    return img_copy


def remove_bright_spots(img, t):
    img_copy = img.copy()
    for p in np.ndindex(img_copy.shape):
        if img_copy[p] > t:
            img_copy[p] = 60
    return img_copy


def median_filter(img, s):
    img_copy = img.copy()
    img_median = ndimage.median_filter(img_copy, size=s)
    return img_median


def mean_filter(img, s):
    img_copy = img.copy()
    img_mean = ndimage.uniform_filter(img_copy, size=s)
    return img_mean


def gaussian_filter(img, s):
    img_copy = img.copy()
    img_gauss = ndimage.gaussian_filter(img_copy, sigma=s)
    return img_gauss


def anisotropic_filter(img):
    img_copy = img.copy()
    img_anisotropic = anisotropic_diffusion(img_copy)
    return img_anisotropic


