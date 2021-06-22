import matplotlib.pyplot as plt
import numpy as np


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
    if pixel[0] - 1 >= 0 and pixel[1] - 1 >= 0 and pixel[0] + 1 <= img.shape[0] - 1 and pixel[1] + 1 <= img.shape[1] - 1:
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
        deviation += (img[filter_neighbors_row, filter_neighbors_col] - mean)**2
    return deviation


def add_border(array):
    array_with_border = np.zeros((array.shape[0]+2, array.shape[1]+2))
    array_with_border[1:array.shape[0]+1, 1:array.shape[1]+1] = array
    return array_with_border

