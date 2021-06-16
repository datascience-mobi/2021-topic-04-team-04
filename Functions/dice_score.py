
import skimage.io as sk
import numpy as np


def segmented_image_clip(segmented_image, background_region):
    """
    image clipping of result image (from own method)
    :param segmented_image: result image from method region growing with region numbers (2D array)
    :param background_region: region number of background of segmented_image (int)
    :return: clipped image where all cells have value 1 and background has value 0 (2D array)
    """
    for i in range(0, segmented_image.shape[0]):
        for j in range(0, segmented_image.shape[1]):
            region = segmented_image[i, j]
            if region == background_region:
                segmented_image[i, j] = 0
            else:
                segmented_image[i, j] = 1
    return segmented_image


def find_background_number(regions):
    """
    determines region number of background as it is the biggest region
    :param regions: segmented image with region number (2D array)
    :return: number of background/ region with maximal number of elements (int)
    """
    regions = regions.astype(int)
    background_number = np.bincount(regions.flatten()).argmax()
    return background_number


def gt_clip(gt):
    """
    image clipping for ground truth image for comparison with result image
    :param gt: ground truth image with region numbers (2D array)
    :return: image where every cells has region value 1 and background has region value 0 (2D array)
    """
    clipped_gt = np.ndarray.clip(gt, 0, 1)
    return clipped_gt


def intersection_count(segmented_image, gt, region_number):
    """
    calculates number of elements that are in both images assigned to a specific region(region_number)
    :param segmented_image: result image with region numbers (2D array)
    :param gt: ground truth image with region numbers (2D array)
    :param region_number: region for which the amount of elements in intersection of both images is calculated (int)
    :return: number of elements that are assigned from both images to region_number (int)
    """
    intersection_number = 0
    for pixel in np.ndindex(segmented_image.shape):
        if segmented_image[pixel] == gt[pixel] == region_number:
            intersection_number += 1
    return intersection_number


def region_count(regions_image, region_number):
    """
    calculates number of elements in a region
    :param regions_image: image with region numbers (2D array)
    :param region_number: region for which number of elements should be counted (int)
    :return: number of elements in the region (int)
    """
    count_region = np.sum(regions_image == region_number)
    return count_region


def region_dice_score(segmented_image, gt, region_number):
    """
    calculates dice score for a region
    :param segmented_image: result image with region numbers (2D array)
    :param gt: ground truth image with region numbers (2D array)
    :param region_number: region for which the dice score is calculated (int)
    :return: dice score for the region (float, between 0 and 1)
    """
    count_intersection = intersection_count(segmented_image, gt, region_number)
    count_segmented_image = region_count(segmented_image, region_number)
    count_gt = region_count(gt, region_number)
    region_score = 2 * count_intersection/(count_segmented_image + count_gt)
    return region_score


def dice_score_weighted(segmented_image, gt):
    """
    calculates weighted dice score looking separately the different regions so all regions have the same impact (regardless the size)
    :param segmented_image: result image with region numbers (2D array)
    :param gt: ground truth image with region numbers (2D array)
    :return: weighted dice score of whole image (float, between 0 and 1)
    """
    dice_score_background = region_dice_score(segmented_image, gt, 0)
    dice_score_nucleus = region_dice_score(segmented_image, gt, 1)
    dice_score = 0.5 * (dice_score_nucleus + dice_score_background)
    return dice_score


def dice_score_unweighted(segmented_image, gt):
    """
    calculates dice score for the whole image without weighing all regions equally
    :param segmented_image: result image with region numbers (2D array)
    :param gt: ground truth image with region numbers (2D array)
    :return: unweighted dice score of whole image (float, between 0 and 1)
    """
    count_intersection = intersection_count(segmented_image, gt, 0) + intersection_count(segmented_image, gt, 1)
    dice_score = 2*count_intersection/(segmented_image.size + gt.size)
    return dice_score


def evaluate_accuracy_weighted(segmented_image, gt):
    """
    calculates weighted dice score of two clipped images (clipping needed for comparability)
    :param segmented_image: result image with region numbers (2D array)
    :param gt: ground truth image with region numbers (2D array)
    :return: wighted dice score (float, between 0 and 1)
    """
    clipped_gt = gt_clip(gt)
    background_number = find_background_number(segmented_image)
    clipped_segmented_image = segmented_image_clip(segmented_image, background_number)
    dice_score = dice_score_weighted(clipped_segmented_image, clipped_gt)
    return dice_score


def evaluate_accuracy_unweighted(segmented_image, gt):
    """
    calculates unweighted dice score of two comparable images (achieved with image clipping)
    :param segmented_image: result image with region numbers (2D array)
    :param gt:  ground truth image with region numbers (2D array)
    :return: unweighted dice score (float, between 0 and 1)
    """
    clipped_gt = gt_clip(gt)
    background_number = find_background_number(segmented_image)
    clipped_segmented_image = segmented_image_clip(segmented_image, background_number)
    dice_score = dice_score_unweighted(clipped_segmented_image, clipped_gt)
    return dice_score


if __name__ == '__main__':
    gt = sk.imread("../Data/N2DH-GOWT1/gt/man_seg01.tif")
    gt_resize = gt[300:350, 400:450]

    segmented_image = sk.imread("../t01tifimg.tif")

    dice_score_weight = evaluate_accuracy_weighted(segmented_image, gt_resize)
    dice_score_unweight = evaluate_accuracy_unweighted(segmented_image, gt_resize)
    print(dice_score_weight)
    print(dice_score_unweight)