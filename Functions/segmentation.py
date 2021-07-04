import skimage.io as sk
from Functions import image_processing as ip
from Functions import region_merging as rm
from Functions import seeded_region_growing as srg
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

    dice_score_unweighted = ds.evaluate_accuracy_unweighted(image_merged.copy(), gt)
    dice_score_weighted = ds.evaluate_accuracy_weighted(image_merged.copy(), gt)
    print("unweighted dice score: " + str(dice_score_unweighted) + ", weighted dice score: " + str(dice_score_weighted))

    return image_merged


if __name__ == '__main__':
    image_intensity = sk.imread("../Data/N2DH-GOWT1/img/t01.tif")
    img_gt = sk.imread("../Data/N2DH-GOWT1/gt/man_seg01.tif")

    image_seeds = sd.seeds(image_intensity, 0.7)
    image_seeds_s = Image.fromarray(image_seeds.copy())
    image_seeds_s.save("../Result_Pictures/Seeded_Region_Growing/N2DH-GOWT1/t01_srg_seeds.tif")

    image_for_srg = sd.seed_merging(image_seeds.copy())
    image_for_srg_reduced = sd.reduce_region_number(image_for_srg.copy(), 2)
    image_srg = srg.region_growing(image_intensity, image_for_srg_reduced.copy())
    image_srg_s = Image.fromarray(image_srg.copy())
    image_srg_s.save("../Result_Pictures/Seeded_Region_Growing/N2DH-GOWT1/t01_srg_srg.tif")

    image_merged = rm.region_merging(image_srg.copy(), image_intensity, 0.05, 400)
    image_merged_s = Image.fromarray(image_merged.copy())
    image_merged_s.save("../Result_Pictures/Seeded_Region_Growing/N2DH-GOWT1/t01_srg_merged.tif")

    image_final = ds.final_clipping(image_merged.copy())
    image_final_s = Image.fromarray(image_final.copy())
    image_final_s.save("../Result_Pictures/Seeded_Region_Growing/N2DH-GOWT1/t01_srg_final.tif")