import skimage.io as sk
import numpy as np
from Functions import image_processing as ip
from Functions import segmentation as seg


def find_background_number(reg):
    """
    determines region number of background as it is the biggest region
    :param reg: segmented image with region number (2D array)
    :return: number of background/ region with maximal number of elements (int)
    """
    reg = reg.astype(int)
    background_number = np.bincount(reg.flatten()).argmax()
    return background_number


def segmented_image_clip(segmented_img, background_region):
    """
    image clipping of result image (from own method)
    :param segmented_img: result image from method region growing with region numbers (2D array)
    :param background_region: region number of background of segmented_image (int)
    :return: clipped image where all cells have value 1 and background has value 0 (2D array)
    """
    for pixel in np.ndindex(segmented_img.shape):
        region = segmented_img[pixel]
        if region == background_region:
            segmented_img[pixel] = 0
        else:
            segmented_img[pixel] = 1
    return segmented_img


def gt_clip(gt):
    """
    image clipping for ground truth image for comparison with result image
    :param gt: ground truth image with region numbers (2D array)
    :return: image where every cells has region value 1 and background has region value 0 (2D array)
    """
    clipped_gt = np.ndarray.clip(gt, 0, 1)
    return clipped_gt


def intersection_count(segmented_img, gt, region_number):
    """
    calculates number of elements that are in both images assigned to a specific region(region_number)
    :param segmented_img: result image with region numbers (2D array)
    :param gt: ground truth image with region numbers (2D array)
    :param region_number: region for which the amount of elements in intersection of both images is calculated (int)
    :return: number of elements that are assigned from both images to region_number (int)
    """
    intersection_number = 0
    for pixel in np.ndindex(segmented_img.shape):
        if segmented_img[pixel] == gt[pixel] == region_number:
            intersection_number += 1
    return intersection_number


def region_count(segmented_img, region_number):
    """
    calculates number of elements in a region
    :param segmented_img: image with region numbers (2D array)
    :param region_number: region for which number of elements should be counted (int)
    :return: number of elements in the region (int)
    """
    count_region = np.sum(segmented_img == region_number)
    return count_region


def dice_score(segmented_img, gt):
    """
    calculates dice score
    :param segmented_img: image after region merging (2d array)
    :param gt: ground truth image (2d array)
    :return: dice score (float between 0 and 1)
    """
    clipped_gt = gt_clip(gt.copy())
    background_number = find_background_number(segmented_img.copy())
    clipped_segmented_image = segmented_image_clip(segmented_img.copy(), background_number)

    true_positive = intersection_count(clipped_segmented_image, clipped_gt, 1)
    true_negative = intersection_count(clipped_segmented_image, clipped_gt, 0)
    number_of_pixels = segmented_img.size

    dice_score_value = 2 * true_positive / (number_of_pixels - true_negative + true_positive)
    return dice_score_value


def final_clipping(segmented_img):
    """
    use segmented image to define nuclei and background
    :param segmented_img: (2d array)
    :return: clipped image (background: 0, nucleus: 1)
    """
    background_number = find_background_number(segmented_img.copy())
    clipped_segmented_image = segmented_image_clip(segmented_img.copy(), background_number)
    return clipped_segmented_image


if __name__ == '__main__':
    image_hela33_small = sk.imread("../Data/N2DL-HeLa/img/t13.tif")[100:200, 450:550]
    image_hela33_small_n = ip.subtract_minimum(image_hela33_small.copy())
    image_gt_hela33_small = sk.imread("../Data/N2DL-HeLa/gt/man_seg13.tif")[100:200, 450:550]
    image_hela13_clipped = ip.image_clipping(image_hela33_small_n, 0.03 * np.amax(image_hela33_small_n),
                                             0.1 * np.amax(image_hela33_small_n))
    image_clipped_segmented_urg = seg.unseeded_segmentation(image_hela13_clipped, image_gt_hela33_small.copy(), (0, 0),
                                                            50, 0.1, 300)

    dice_value = dice_score(image_clipped_segmented_urg, image_gt_hela33_small)
    print(dice_value)