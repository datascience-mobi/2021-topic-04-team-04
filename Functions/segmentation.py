import skimage.io as sk
import numpy as np
from Functions import image_processing as ip
from Functions import region_merging as rm
from Functions import seeded_region_growing as srg
from Functions import unseeded_region_growing as urg
from Functions import seed_detection as sd
from Functions import dice_score as ds
from PIL import Image


def seeded_segmentation(img, gt, threshold_seeds, threshold_merging_intensity, threshold_merging_size):
    """
    total seeded region growing algorithm
    :param img: intensity values (2d array)
    :param gt: (2d array)
    :param threshold_seeds: relative euclidean distance 0 < threshold < 1 (float)
    :param threshold_merging_intensity: threshold for intensity distance between region (float, 0 < threshold < 1)
    :param threshold_merging_size: threshold for region size (int)
    :return: resulting image (2d array)
    :return: unweighted dice score (float)
    :return: weighted dice score (float)
    """
    image_seeds = sd.seeds(img, threshold_seeds)
    image_regions_from_seeds = sd.seed_merging(image_seeds)
    image_srg = srg.region_growing(img, image_regions_from_seeds)
    image_merged = rm.region_merging(image_srg, img, threshold_merging_intensity, threshold_merging_size)
    image_clipped = ds.final_clipping(image_merged)

    dice_score_unweighted = ds.evaluate_accuracy_unweighted(image_merged.copy(), gt)
    dice_score_weighted = ds.evaluate_accuracy_weighted(image_merged.copy(), gt)
    print("unweighted dice score: " + str(dice_score_unweighted) + ", weighted dice score: " + str(dice_score_weighted))

    return image_clipped


def unseeded_segmentation(img, gt, start_pixel, threshold_region_growing, threshold_merging_intensity,
                          threshold_merging_size):
    """
    total unseeded region growing algorithm
    :param img: intensity values (2d array)
    :param gt: (2d array)
    :param start_pixel: start pixel for unseeded region growing (tuple)
    :param threshold_region_growing: threshold for region growing (float/int)
    :param threshold_merging_intensity: threshold for intensity distance between region (float, 0 < threshold < 1)
    :param threshold_merging_size: threshold for region size (int)
    :return: resulting image (2d array)
    """
    image_urg = urg.unseeded_region_growing_algorithm(img, start_pixel, threshold_region_growing)
    image_merged = rm.region_merging(image_urg, img, threshold_merging_intensity, threshold_merging_size)
    image_filtered = ip.median_filter(image_merged, 3)
    image_clipped = ds.final_clipping(image_filtered)

    dice_score_unweighted = ds.evaluate_accuracy_unweighted(image_filtered.copy(), gt)
    dice_score_weighted = ds.evaluate_accuracy_weighted(image_filtered.copy(), gt)

    print("unweighted dice score: " + str(dice_score_unweighted) + ", weighted dice score: " + str(dice_score_weighted))

    return image_clipped


if __name__ == '__main__':
    image_intensity = sk.imread("../Data/N2DL-HeLa/img/t13.tif")
    image_intensity = ip.subtract_minimum(image_intensity)
    image_intensity = ip.image_clipping(image_intensity, 0.05 * np.amax(image_intensity), 0.15 * np.amax(image_intensity))
    image_intensity = ip.median_filter(image_intensity, 3)
    #  image_intensity = ip.remove_bright_spots_with_border(image_intensity, 130, 60, 40)
    img_gt = sk.imread("../Data/N2DL-HeLa/gt/man_seg13.tif")

    use_image_seeds = sd.seeds(image_intensity, 0.05)
    image_seeds_s = Image.fromarray(use_image_seeds.copy())
    image_seeds_s.save("../Result_Pictures/Seeded_Region_Growing/N2DL-HeLa/t13_srg_seeds.tif")

    image_for_srg = sd.seed_merging(use_image_seeds.copy())
    image_for_srg_reduced = sd.reduce_region_number(image_for_srg.copy(), 20)
    image_for_srg_reduced_s = Image.fromarray(image_for_srg_reduced.copy())
    image_for_srg_reduced_s.save("../Result_Pictures/Seeded_Region_Growing/N2DL-HeLa/t13_srg_seeds_reduced.tif")

    use_image_srg = srg.region_growing(image_intensity, image_for_srg_reduced.copy())
    image_srg_s = Image.fromarray(use_image_srg.copy())
    image_srg_s.save("../Result_Pictures/Seeded_Region_Growing/N2DL-HeLa/t13_srg_srg.tif")

    use_image_merged = rm.region_merging(use_image_srg.copy(), image_intensity, 0.1, 100)
    image_merged_s = Image.fromarray(use_image_merged.copy())
    image_merged_s.save("../Result_Pictures/Seeded_Region_Growing/N2DL-HeLa/t13_srg_merged.tif")

    image_final = ds.final_clipping(use_image_merged.copy())
    image_final_s = Image.fromarray(image_final.copy())
    image_final_s.save("../Result_Pictures/Seeded_Region_Growing/N2DL-HeLa/t13_srg_clipped.tif")
