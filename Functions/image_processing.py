import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from medpy.filter.smoothing import anisotropic_diffusion


def show_two_images_title(img1, img2, f_title, img1_title, img2_title, colorbar_size):
    plt.figure(figsize=(10, 4))
    plt.suptitle(f_title, fontsize=16)

    s1 = plt.subplot(1, 2, 1)
    s1.set_title(img1_title, fontsize=14)
    plt.imshow(img1, "gray")
    plt.colorbar(shrink=colorbar_size)

    s2 = plt.subplot(1, 2, 2)
    s2.set_title(img2_title, fontsize=14)
    plt.imshow(img2, "gray")
    plt.colorbar(shrink=colorbar_size)


def show_three_images_title(img1, img2, img3, f_title, img1_title, img2_title, img3_title, colorbar_size,
                            title_position):
    plt.figure(figsize=(15, 8))
    plt.suptitle(f_title, fontsize=18, y=title_position)

    s1 = plt.subplot(1, 3, 1)
    s1.set_title(img1_title, fontsize=16)
    plt.imshow(img1, "gray")
    plt.colorbar(shrink=colorbar_size)

    s2 = plt.subplot(1, 3, 2)
    s2.set_title(img2_title, fontsize=16)
    plt.imshow(img2, "gray")
    plt.colorbar(shrink=colorbar_size)

    s3 = plt.subplot(1, 3, 3)
    s3.set_title(img3_title, fontsize=16)
    plt.imshow(img3, "gray")
    plt.colorbar(shrink=colorbar_size)


def show_four_images_title(img1, img2, img3, img4, f_title, img1_title, img2_title, img3_title, img4_title,
                           colorbar_size):
    plt.figure(figsize=(15, 8))
    plt.suptitle(f_title, fontsize=16, y=0.7)

    s1 = plt.subplot(1, 4, 1)
    s1.set_title(img1_title, fontsize=14)
    plt.imshow(img1, "gray")
    plt.colorbar(shrink=colorbar_size)

    s2 = plt.subplot(1, 4, 2)
    s2.set_title(img2_title, fontsize=14)
    plt.imshow(img2, "gray")
    plt.colorbar(shrink=colorbar_size)

    s3 = plt.subplot(1, 4, 3)
    s3.set_title(img3_title, fontsize=14)
    plt.imshow(img3, "gray")
    plt.colorbar(shrink=colorbar_size)

    s4 = plt.subplot(1, 4, 4)
    s4.set_title(img4_title, fontsize=14)
    plt.imshow(img4, "gray")
    plt.colorbar(shrink=colorbar_size)


def show_six_images_two_rows_title(img1, img2, img3, img4, img5, img6, f_title, img1_title, img2_title, img3_title,
                                   img4_title, img5_title, img6_title,
                                   colorbar_size):
    plt.figure(figsize=(15, 7))
    plt.suptitle(f_title, fontsize=16, y=0.95)

    s1 = plt.subplot(2, 3, 1)
    s1.set_title(img1_title, fontsize=14)
    plt.imshow(img1, "gray")
    plt.colorbar(shrink=colorbar_size)

    s2 = plt.subplot(2, 3, 2)
    s2.set_title(img2_title, fontsize=14)
    plt.imshow(img2, "gray")
    plt.colorbar(shrink=colorbar_size)

    s3 = plt.subplot(2, 3, 3)
    s3.set_title(img3_title, fontsize=14)
    plt.imshow(img3, "gray")
    plt.colorbar(shrink=colorbar_size)

    s4 = plt.subplot(2, 3, 4)
    s4.set_title(img4_title, fontsize=14)
    plt.imshow(img4, "gray")
    plt.colorbar(shrink=colorbar_size)

    s5 = plt.subplot(2, 3, 5)
    s5.set_title(img5_title, fontsize=14)
    plt.imshow(img5, "gray")
    plt.colorbar(shrink=colorbar_size)

    s6 = plt.subplot(2, 3, 6)
    s6.set_title(img6_title, fontsize=14)
    plt.imshow(img6, "gray")
    plt.colorbar(shrink=colorbar_size)


def show_six_images_title(img1, img2, img3, img4, img5, img6, f_title, img1_title, img2_title, img3_title,
                          img4_title, img5_title, img6_title,
                          colorbar_size):
    plt.figure(figsize=(20, 8))
    plt.suptitle(f_title, fontsize=16, y=0.7)

    s1 = plt.subplot(1, 6, 1)
    s1.set_title(img1_title, fontsize=14)
    plt.imshow(img1, "gray")
    plt.colorbar(shrink=colorbar_size)

    s2 = plt.subplot(1, 6, 2)
    s2.set_title(img2_title, fontsize=14)
    plt.imshow(img2, "gray")
    plt.colorbar(shrink=colorbar_size)

    s3 = plt.subplot(1, 6, 3)
    s3.set_title(img3_title, fontsize=14)
    plt.imshow(img3, "gray")
    plt.colorbar(shrink=colorbar_size)

    s4 = plt.subplot(1, 6, 4)
    s4.set_title(img4_title, fontsize=14)
    plt.imshow(img4, "gray")
    plt.colorbar(shrink=colorbar_size)

    s5 = plt.subplot(1, 6, 5)
    s5.set_title(img5_title, fontsize=14)
    plt.imshow(img5, "gray")
    plt.colorbar(shrink=colorbar_size)

    s6 = plt.subplot(1, 6, 6)
    s6.set_title(img6_title, fontsize=14)
    plt.imshow(img6, "gray")
    plt.colorbar(shrink=colorbar_size)


def show_images_side_by_side(img1, img2):
    f = plt.figure(figsize=(12, 7))
    f.add_subplot(1, 2, 1)
    plt.imshow(img1, "gray")

    f.add_subplot(1, 2, 2)
    plt.imshow(img2, "gray")

    plt.show()


def show_three_images_colorbar(img1, img2, img3, colorbar_size):
    f = plt.figure(figsize=(15, 8))

    f.add_subplot(1, 3, 1)
    plt.imshow(img1, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(1, 3, 2)
    plt.imshow(img2, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(1, 3, 3)
    plt.imshow(img3, "gray")
    plt.colorbar(shrink=colorbar_size)

    plt.show()


def show_six_images_two_rows(img1, img2, img3, img4, img5, img6, colorbar_size):
    f = plt.figure(figsize=(15, 6))

    f.add_subplot(2, 3, 1)
    plt.imshow(img1, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(2, 3, 2)
    plt.imshow(img2, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(2, 3, 3)
    plt.imshow(img3, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(2, 3, 4)
    plt.imshow(img4, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(2, 3, 5)
    plt.imshow(img5, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(2, 3, 6)
    plt.imshow(img6, "gray")
    plt.colorbar(shrink=colorbar_size)

    plt.show()


def show_four_images_colorbar(img1, img2, img3, img4, colorbar_size):
    f = plt.figure(figsize=(17, 8))

    f.add_subplot(1, 4, 1)
    plt.imshow(img1, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(1, 4, 2)
    plt.imshow(img2, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(1, 4, 3)
    plt.imshow(img3, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(1, 4, 4)
    plt.imshow(img4, "gray")
    plt.colorbar(shrink=colorbar_size)

    plt.show()


def show_six_images_colorbar(img1, img2, img3, img4, img5, img6, colorbar_size):
    f = plt.figure(figsize=(22, 8))

    f.add_subplot(1, 6, 1)
    plt.imshow(img1, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(1, 6, 2)
    plt.imshow(img2, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(1, 6, 3)
    plt.imshow(img3, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(1, 6, 4)
    plt.imshow(img4, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(1, 6, 5)
    plt.imshow(img5, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(1, 6, 6)
    plt.imshow(img6, "gray")
    plt.colorbar(shrink=colorbar_size)

    plt.show()


def show_two_images(img1, img2):
    f = plt.figure(figsize=(15, 8))

    f.add_subplot(1, 2, 1)
    plt.imshow(img1, "gray")

    f.add_subplot(1, 2, 2)
    plt.imshow(img2, "gray")

    plt.show()


def show_two_images_colorbar(img1, img2, colorbar_size):
    f = plt.figure(figsize=(15, 8))

    f.add_subplot(1, 2, 1)
    plt.imshow(img1, "gray")
    plt.colorbar(shrink=colorbar_size)

    f.add_subplot(1, 2, 2)
    plt.imshow(img2, "gray")
    plt.colorbar(shrink=colorbar_size)

    plt.show()


def show_image(img, x, y):
    """
    shows images with a specific size, in gray and with a colorbar
    :param img: image to be shown (2D array)
    :param x: length of image in x-direction (int)
    :param y: length of image in y-direction (int)
    :return: picture of image in jupyter notebook
    """
    plt.figure(figsize=(x, y))
    plt.imshow(img, "gray")
    plt.colorbar()


def img_resize(img, size_x, size_y):
    """
    returns a smaller cut from the upper right corner of an image
    :param img: image to be cut from (2D array)
    :param size_x: length of image in x-direction (int)
    :param size_y: length of image in y-direction (int)
    :return: created image (2D array)
    """
    new_size = [size_y, size_x]
    img_new = np.zeros(new_size)
    for x in np.ndindex(new_size[0], new_size[1]):
        img_new[x] = img[x]
    return img_new


def t_pic():
    """
    creats a black image with white spots
    :return: black image with white spots (2D array)
    """
    img = np.zeros([100, 100])
    img[30:40, 30:40] = 1
    img[75:80, 75:80] = 1
    return img


def is_border_pixel(pixel, img):
    """
    tests, whether a pixel is a border pixel
    :param pixel: tested pixel (tuple)
    :param img: image of which the border is tested (2D array)
    :return: True for border pixel, False for not border pixel (True/False)
    """
    if pixel[0] - 1 >= 0 and pixel[1] - 1 >= 0 and pixel[0] + 1 <= img.shape[0] - 1 and pixel[1] + 1 <= img.shape[
        1] - 1:
        return False
    return True


def filter_iteration_sum(img, pixel, size):
    """
    calculates the sum of the intensity of all pixels in a filter mask
    :param img: image with intensity values (2D array)
    :param pixel: pixel in the middle of the mask (tuple)
    :param size: size of the mask (int)
    :return: sum of intensity values (float)
    """
    n = (size - 1) // 2
    neighborhood_sum = 0
    for filter_pixel in np.ndindex(size, size):
        filter_neighbors_row = pixel[0] + filter_pixel[0] - n
        filter_neighbors_col = pixel[1] + filter_pixel[1] - n
        neighborhood_sum += img[filter_neighbors_row, filter_neighbors_col]
    return neighborhood_sum


def filter_iteration_deviation(img, pixel, size, mean):
    """
    calculates sum of deviations of the mean value of the intensity values of all pixels in a filter mask
    :param img: image with intensity values (2D array)
    :param pixel: pixel in the middle of the mask (tuple)
    :param size: size of the mask (int)
    :param mean: mean value of the intensity values of the pixels in the mask (float)
    :return: sum of deviations from the mean (float)
    """
    n = (size - 1) // 2
    deviation = 0
    for filter_pixel in np.ndindex(size, size):
        filter_neighbors_row = pixel[0] + filter_pixel[0] - n
        filter_neighbors_col = pixel[1] + filter_pixel[1] - n
        deviation += (img[filter_neighbors_row, filter_neighbors_col] - mean) ** 2
    return deviation


def add_border(array):
    """
    adds a border with zeros to an array
    :param array: array that should get a border (2D array)
    :return: array with added border of zeros (2D array)
    """
    array_with_border = np.zeros((array.shape[0] + 2, array.shape[1] + 2))
    array_with_border[1:array.shape[0] + 1, 1:array.shape[1] + 1] = array
    return array_with_border


def add_border_variable(array, border_size):
    array_with_border = np.zeros((array.shape[0] + 2 * border_size, array.shape[1] + 2 * border_size))
    array_with_border[border_size:array.shape[0] + border_size, border_size:array.shape[1] + border_size] = array
    return array_with_border


def remove_border_variable(array, border_size):
    array = np.delete(array, np.s_[0:border_size], axis=1)
    array = np.delete(array, np.s_[0:border_size], axis=0)
    array = np.delete(array, np.s_[array.shape[1] - border_size:array.shape[1]], axis=1)
    array = np.delete(array, np.s_[array.shape[0] - border_size:array.shape[0]], axis=0)
    return array


def remove_border(array):
    """
    removes the border of an array
    :param array: array which border should be removed (2D array)
    :return: array without its border (2D array)
    """
    array = np.delete(array, 0, axis=1)
    array = np.delete(array, array.shape[1] - 1, axis=1)
    array = np.delete(array, 0, axis=0)
    array = np.delete(array, array.shape[0] - 1, axis=0)
    return array


def histogram(img):
    """
    creates a histogram of the intensity values of an image
    :param img: image with intensity values (2D array)
    :return: shown plot of the histogram in a jupyter notebook
    """
    histo = plt.hist(img.flatten(), log=True)
    plt.show(histo)


def image_clipping(img, t1, t2):
    """
    copies the image and performs image clipping the copied one
    :param img: image with intensity values (2D array)
    :param t1: lower value for image clipping (int/float)
    :param t2: higher value for image clipping (int/float)
    :return: clipped image (2D array)
    """
    img_copy = img.copy()
    clipped_img = np.clip(img_copy, t1, t2)
    return clipped_img


def image_clipping_extreme(img, t1, t2):
    """
    copies the image and sets values below the first threshold to zero
    values above the second threshold are set to the maximum intensity
    :param img: image with intensity values (2D array)
    :param t1: first threshold (lower) (int/float)
    :param t2: second threshold (upper) (int/float)
    :return: image with changed intensity values (2D array)
    """
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


def remove_bright_spots(img, t_bright, t_background):
    """
    copies the image and tries to remove extremely bright spots for a picture
    sets all the pixels above the first threshold to the value of the second threshold
    :param t_bright: pixels above this threshold are considered as to bright (int/float)
    :param t_background: pixels are set to this intensity value (int/float)
    :param img: image with intensity values (2D array)
    :return: image with changed intensity values (2D array)
    """
    img_copy = img.copy()
    pos_bright_spot = np.where(img_copy > t_bright)
    img_copy[pos_bright_spot[0], pos_bright_spot[1]] = t_background
    return img_copy


def remove_bright_spots_with_border(img, t_bright, t_background, t_border):
    """
    copies the image and tries to remove extremely bright spots for a picture
    sets all the pixels above the first threshold to the value of the second threshold
    :param t_bright: pixels above this threshold are considered as to bright (int/float)
    :param t_border: number of border pixel (int)
    :param t_background: pixels are set to this intensity value (int/float)
    :param img: image with intensity values (2D array)
    :return: image with changed intensity values (2D array)
    """
    img_copy = img.copy()
    img_copy = add_border_variable(img_copy, t_border)
    pos_bright_spot = np.where(img_copy > t_bright)
    img_copy[pos_bright_spot[0], pos_bright_spot[1]] = t_background
    for border_number in range(1, t_border + 1):
        img_copy[pos_bright_spot[0] - border_number, pos_bright_spot[1]] = t_background
        img_copy[pos_bright_spot[0] + border_number, pos_bright_spot[1]] = t_background
        img_copy[pos_bright_spot[0], pos_bright_spot[1] - border_number] = t_background
        img_copy[pos_bright_spot[0], pos_bright_spot[1] + border_number] = t_background
    img_copy = remove_border_variable(img_copy, t_border)
    return img_copy


def median_filter(img, s):
    """
    copies image and performs median filtering on the image
    :param img: image with intensity values (2D array)
    :param s: size of the filter mask (int)
    :return: filtered image (2D array)
    """
    img_copy = img.copy()
    img_median = ndimage.median_filter(img_copy, size=s)
    return img_median


def mean_filter(img, s):
    """
    copies image and performs mean filtering on the image
    :param img: image with intensity values (2D array)
    :param s: size of the filter mask (int)
    :return: filtered image (2D array)
    """
    img_copy = img.copy()
    img_mean = ndimage.uniform_filter(img_copy, size=s)
    return img_mean


def gaussian_filter(img, s):
    """
    copies image and performs gaussian filtering on the image
    :param img: image with intensity values (2D array)
    :param s: size of the filter mask (int)
    :return: filtered image (2D array)
    """
    img_copy = img.copy()
    img_gauss = ndimage.gaussian_filter(img_copy, sigma=s)
    return img_gauss


def anisotropic_filter(img):
    """
    copies image and performs anisotropic filtering on the image
    :param img: image with intensity values (2D array)
    :return: filtered image (2D array)
    """
    img_copy = img.copy()
    img_anisotropic = anisotropic_diffusion(img_copy)
    return img_anisotropic


def normalize_intensity(img):
    img_normalized = img.copy() / np.amax(img)
    return img_normalized


def subtract_minimum(img):
    img_subtracted = img.copy() - np.amin(img)
    return img_subtracted
