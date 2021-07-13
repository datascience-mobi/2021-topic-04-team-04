import numpy as np
import skimage.io as sk
import matplotlib.pyplot as plt
from Functions import image_processing as ip
from Functions import seeded_region_growing as srg
from Functions import unseeded_region_growing as urg
from Functions import seed_detection as sd
from Functions import region_merging as rm
from Functions import old_seeded_region_growing as old_srg
from Functions import old_unseeded_region_growing as old_urg
from Functions import dice_score as ds
from Functions import segmentation as seg
import pandas as pd
import seaborn as sns


def barplot_preprocessing():
    """
    creates barplot for the dice scores of the evaluation of the preprocessing
    :return: barplot for the preprocessing
    """
    columns_names = ["Dice Score", "Segmentation Method", "Preprocessing"]
    dice_score = [0.908, 0.945, 0.983, 0.984, 0.980, 0.979, 0.911, 0.984, 0.908, 0.962]
    segmentation_method = ["Seeded", "Unseeded", "Seeded", "Unseeded", "Seeded", "Unseeded", "Seeded", "Unseeded",
                           "Seeded", "Unseeded"]
    preprocessing = ["unprocessed", "unprocessed", "clipped", 'clipped', 'extreme clipped', 'extreme clipped', 'median',
                     'median', 'gauss', 'gauss']

    df = pd.DataFrame(list(zip(dice_score, segmentation_method, preprocessing)), columns=columns_names)

    ax = sns.barplot(x="Preprocessing", y="Dice Score", hue="Segmentation Method", data=df, palette="dark")
    ax.set(ylim=(0.85, 1))
    plt.legend(loc=3)


def barplot_results():
    """
    creates a barplot for the dice scores of the segmented images of the different data sets
    :return: barplot for the different data sets
    """
    columns_names = ["Dice Score", "Segmentation Method", "Data Sets"]
    dice_score = [0.928, 0.929, 0.878, 0.928, 0.68, 0.738]
    segmentation_method = ["Seeded", "Unseeded", "Seeded", "Unseeded", "Seeded", "Unseeded"]
    data_sets = ["N2DH-GOWT1", "N2DH-GOWT1", "N2DL-HeLa", "N2DL-HeLa", "NIH3T3", "NIH3T3"]

    df = pd.DataFrame(list(zip(dice_score, segmentation_method, data_sets)), columns=columns_names)

    ax = sns.barplot(x="Data Sets", y="Dice Score", hue="Segmentation Method", data=df, palette="dark")
    ax.set(ylim=(0.65, 1))
    plt.legend(loc=3)


def barplot_runtime():
    columns_names = ["Runtime in ms", "Sd_runtime in ms", "Segmentation Method", "Algorithm version"]
    runtime = [6440, 18200, 703, 672]
    sd_time = [99.1, 796, 11.1, 24.4]
    segmentation_method = ["Seeded", "Unseeded", "Seeded", "Unseeded"]
    version = ["Old", "Old", "New", "New"]

    df = pd.DataFrame(list(zip(runtime, sd_time, segmentation_method, version)), columns=columns_names)
    #  print(df)

    ax = sns.barplot(x="Algorithm version", y="Runtime in ms", hue="Segmentation Method", data=df, palette="dark")
    ax.set(ylim=(0, 20000))
    plt.legend(loc=3)

def load_images():
    """
    loads example images of all the data sets
    :return: graph with three images, one of each data set
    """
    image_intensity = sk.imread("Data/N2DH-GOWT1/img/t01.tif")
    image_intensity_data2 = sk.imread("Data/N2DL-HeLa/img/t52.tif")
    image_intensity_data3 = sk.imread("Data/NIH3T3/img/dna-42.png")
    ip.show_three_images_colorbar(image_intensity, image_intensity_data2, image_intensity_data3, 0.45)


def clipping_examples():
    """
    clips images using different methods and shows the results in a graph
    :return: graph with the original image, the clipped one and the extremly clipped one
    """
    image_intensity = sk.imread("Data/N2DH-GOWT1/img/t01.tif")
    img_clipped = ip.image_clipping(image_intensity, 5, 25)
    img_clipped_extreme = ip.image_clipping_extreme(image_intensity, 5, 15)
    ip.show_three_images_colorbar(image_intensity, img_clipped, img_clipped_extreme, 0.45)


def bright_spots_example():
    """
    removed bright spots using different methods and shows the results
    :return: graph with the original image, removed bright spot image and removed bright spot with border image
    """
    img_with_bright_spots = sk.imread("Data/NIH3T3/img/dna-33.png")
    img_removed_spots = ip.remove_bright_spots(img_with_bright_spots, 200, 60)
    img_removed_spots_with_border = ip.remove_bright_spots_with_border(img_with_bright_spots, 200, 60, 40)
    ip.show_three_images_colorbar(img_with_bright_spots, img_removed_spots, img_removed_spots_with_border, 0.35)


def region_growing_example():
    """
    performs seeded and unseeded region growing on a smaller image and shows the different steps and the results
    :return: the first for images shown are for seeded region growing: the found regions, the merging by similarity,
    the merging by size and the clipped image. The second for images show the same steps for unseeded region growing.
    """
    image_intensity = sk.imread("Data/N2DH-GOWT1/img/t01.tif")
    image_intensity_small = image_intensity[300:400, 400:500]
    image_seeds = sd.seeds(image_intensity_small, 0.5)
    image_regions_from_seeds = sd.seed_merging(image_seeds)

    image_regions = srg.region_growing(image_intensity_small, image_regions_from_seeds.copy())
    image_result_unseeded = urg.unseeded_region_growing_algorithm(image_intensity_small, (0, 0), 5)

    results_region_merging_similarity = rm.distance_merging_while(image_regions.copy(), 0.05, image_intensity_small)
    image_rm_similarity, inter_region_neighbors, means = results_region_merging_similarity
    image_rm_size = rm.region_merging_size(image_intensity_small, image_rm_similarity.copy(), inter_region_neighbors,
                                           means, 500)
    image_clipped_s = ds.final_clipping(image_rm_size.copy())
    ip.show_four_images_colorbar(image_regions, image_rm_similarity, image_rm_size, image_clipped_s, 0.38)

    results_region_merging_similarity_urg = rm.distance_merging_while(image_result_unseeded.copy(), 0.08,
                                                                      image_intensity_small)
    image_rm_similarity_urg, inter_region_neighbors_urg, means_urg = results_region_merging_similarity_urg
    image_rm_size_urg = rm.region_merging_size(image_intensity_small, image_rm_similarity_urg.copy(),
                                               inter_region_neighbors_urg, means_urg, 1000)
    image_clipped_s_urg = ds.final_clipping(image_rm_size_urg.copy())

    ip.show_four_images_colorbar(image_result_unseeded, image_rm_similarity_urg, image_rm_size_urg, image_clipped_s_urg,
                                 0.38)


def results_gowt1_seeded():
    """
    shows the result for seeded region growing on a complete image of the N2DH-GOWT1 data set
    :return: shows four images: the original one, the found regions, the clipped one and the ground truth image
    """
    image_intensity = sk.imread("Data/N2DH-GOWT1/img/t01.tif")
    image_srg_t01 = sk.imread("Result_Pictures/Seeded_Region_Growing/N2DH-GOWT1/t01_srg_srg.tif")
    image_srg_t01_merged_clipped = sk.imread("Result_Pictures/Seeded_Region_Growing/N2DH-GOWT1/t01_srg_final.tif")
    image_gt_t01 = sk.imread("Data/N2DH-GOWT1/gt/man_seg01.tif")
    image_gt_t01 = ds.final_clipping(image_gt_t01)

    ip.show_four_images_colorbar(image_intensity, image_srg_t01, image_srg_t01_merged_clipped, image_gt_t01, 0.38)


def results_gowt1_unseeded():
    """
    shows the impact of median filtering after unseeded region growing
    :return: shows four images: the image after merging, the image clipped directly after merging, the image merged
    and filtered, the image clipped after merging and filtering
    """
    image_urg_t01_merged = sk.imread(
        "Result_Pictures/Unseeded_Region_Growing/N2DH-GOWT1/urg_t01_merged_0.001_10000.tif")
    image_urg_t01_merged_filtered = sk.imread(
        "Result_Pictures/Unseeded_Region_Growing/N2DH-GOWT1/urg_t01_merged_0.001_10000_filtered_3.tif")
    image_urg_t01_merged_clipped = sk.imread(
        "Result_Pictures/Unseeded_Region_Growing/N2DH-GOWT1/urg_t01_clipped.tif")
    image_urg_t01_merged_filtered_clipped = sk.imread(
        "Result_Pictures/Unseeded_Region_Growing/N2DH-GOWT1/urg_t01_filtered_median3_clipped.tif")

    ip.show_four_images_colorbar(image_urg_t01_merged, image_urg_t01_merged_clipped, image_urg_t01_merged_filtered,
                                 image_urg_t01_merged_filtered_clipped, 0.38)


def results_hela():
    """
    shows the results of seeded and unseeded region growing on a complete image of the N2DL-HeLa data set
    :return: shows six images in two rows, the original image, the found region in seeded region growing, the seeded
    image clipped, the unseeded region growing, the unseeded image clipped and the ground truth image
    also prints the dice scores of seeded and unseeded region growing
    """
    image = sk.imread("Data/N2DL-HeLa/img/t13.tif")
    gt = sk.imread("Data/N2DL-HeLa/gt/man_seg13.tif")
    image_srg = sk.imread("Result_Pictures/Seeded_Region_Growing/N2DL-HeLa/t13_srg_srg.tif")
    image_srg_clipped = sk.imread("Result_Pictures/Seeded_Region_Growing/N2DL-HeLa/t13_srg_clipped.tif")
    image_urg = sk.imread("Result_Pictures/Unseeded_Region_Growing/N2DL-HeLa/urg_t13_50.tif")
    image_urg_filtered_clipped = sk.imread("Result_Pictures/Unseeded_Region_Growing/N2DL-HeLa/urg_t13_clipped.tif")
    ip.show_six_images_two_rows(image, image_srg, image_srg_clipped, image_urg, image_urg_filtered_clipped, gt, 0.84)
    dice_score_weighted_srg = ds.evaluate_accuracy_weighted(image_srg_clipped, gt)
    dice_score_unweighted_srg = ds.evaluate_accuracy_unweighted(image_srg_clipped, gt)
    print("Seeded: weighted dice score: " + str(dice_score_weighted_srg) + ", unweighted dice score: " +
          str(dice_score_unweighted_srg))
    dice_score_weighted_urg = ds.evaluate_accuracy_weighted(image_urg_filtered_clipped, gt)
    dice_score_unweighted_urg = ds.evaluate_accuracy_unweighted(image_urg_filtered_clipped, gt)
    print("Unseeded: weighted dice score: " + str(dice_score_weighted_urg) + ", unweighted dice score: " + str(
        dice_score_unweighted_urg))


def nih3t3_show_bright_spots():
    """
    shows the removal of a bright spot on a smaller picture of the NIH3T3 data set
    :return: three images are shown, the original one, one with normally removed bright spot and one with removed
    bright spot with border
    """
    image_intensity_d3_small = sk.imread("Data/NIH3T3/img/dna-42.png")[730:850, 1200:1300]
    img_removed_spots_intact_nuclei = ip.remove_bright_spots(image_intensity_d3_small, 130, 60)
    img_removed_spots_with_border = ip.remove_bright_spots_with_border(image_intensity_d3_small, 130, 60, 30)
    ip.show_three_images_colorbar(image_intensity_d3_small, img_removed_spots_intact_nuclei,
                                  img_removed_spots_with_border, 0.55)


def nih3t3_srg_bright_spots():
    """
    shows the seeded region growing algorithm performed on smaller images of the NIH3T3 data set with different methods
    of bright spot removal
    :return: shows three images, shows the result pictures of the seeded region growing algorithm
    """
    image_intensity_d3_small = sk.imread("Data/NIH3T3/img/dna-42.png")[730:850, 1200:1300]
    gt_data3_small = sk.imread("Data/NIH3T3/gt/42.png")[730:850, 1200:1300]
    img_removed_spots_intact_nuclei = ip.remove_bright_spots(image_intensity_d3_small, 130, 60)
    img_removed_spots_with_border = ip.remove_bright_spots_with_border(image_intensity_d3_small, 130, 60, 30)

    result_seg_with_bright_spots = seg.seeded_segmentation(image_intensity_d3_small, gt_data3_small, 0.05, 0.1, 400)
    result_seg_removed_spots_intact_nuclei = seg.seeded_segmentation(img_removed_spots_intact_nuclei, gt_data3_small,
                                                                     0.05, 0.26, 400)
    result_seg_removed_spots_with_border = seg.seeded_segmentation(img_removed_spots_with_border, gt_data3_small, 0.05,
                                                                   0.2, 400)
    ip.show_three_images_colorbar(result_seg_with_bright_spots, result_seg_removed_spots_intact_nuclei,
                                  result_seg_removed_spots_with_border, 0.55)


def nih3t3_blurs():
    """
    shows the impact of blurs in the NIH3T3 data set on seeded region growing
    :return: shows three images, the original one, the found regions and the clipped one
    """
    image_dna33_small = sk.imread("Data/NIH3T3/img/dna-33.png")[200:500, 900:1200]
    srg_image_dna33_small = sk.imread("Result_Pictures/Unseeded_Region_Growing/NIH3T3/dna-33_5_blurs.tif")
    merged_image_small = rm.region_merging(srg_image_dna33_small.copy(), image_dna33_small, 0.01, 17500)
    clipped_image_small = ds.final_clipping(merged_image_small)
    ip.show_three_images_colorbar(image_dna33_small, srg_image_dna33_small, clipped_image_small, 0.45)


def show_preprocessing():
    """
    shows the different preprocessing methods performed on a smaller image of the N2DL-HeLa data set and their impact
    in seeded and unseeded region growing
    :return: shows three sets of six images
    first set: image normalized, clipped, clipped extreme, median, gauss and ground truth
    second set: seeded region growing on the images of the first set and the ground truth image
    thirs set: unseeded region growing on the images of the first set and the ground truth image
    """
    image_hela33_small = sk.imread("Data/N2DL-HeLa/img/t13.tif")[100:200, 450:550]
    image_hela33_small_n = ip.subtract_minimum(image_hela33_small.copy())
    image_gt_hela33_small = sk.imread("Data/N2DL-HeLa/gt/man_seg13.tif")[100:200, 450:550]

    image_hela33_clipped = ip.image_clipping(image_hela33_small_n, 0.03 * np.amax(image_hela33_small_n),
                                             0.1 * np.amax(image_hela33_small_n))
    image_hela33_clipped_extreme = ip.image_clipping_extreme(image_hela33_small_n, 0.03 * np.amax(image_hela33_small_n),
                                                             0.1 * np.amax(image_hela33_small_n))
    image_hela33_median = ip.median_filter(image_hela33_small_n, 3)
    image_hela33_gauss = ip.gaussian_filter(image_hela33_small_n, 1)
    ip.show_six_images_colorbar(image_hela33_small_n, image_hela33_clipped, image_hela33_clipped_extreme,
                                image_hela33_median, image_hela33_gauss, image_gt_hela33_small, 0.32)

    image_small_segmented = seg.seeded_segmentation(image_hela33_small_n, image_gt_hela33_small, 0.9, 0.1, 300)
    image_clipped_segmented = seg.seeded_segmentation(image_hela33_clipped, image_gt_hela33_small, 0.9, 0.1, 300)
    image_clipped_extreme_segmented = seg.seeded_segmentation(image_hela33_clipped_extreme, image_gt_hela33_small, 0.9,
                                                              0.1, 300)
    image_median_segmented = seg.seeded_segmentation(image_hela33_median, image_gt_hela33_small, 0.9, 0.1, 300)
    image_gauss_segmented = seg.seeded_segmentation(image_hela33_gauss, image_gt_hela33_small, 0.9, 0.1, 300)
    ip.show_six_images_colorbar(image_small_segmented, image_clipped_segmented, image_clipped_extreme_segmented,
                                image_median_segmented, image_gauss_segmented, image_gt_hela33_small, 0.32)

    image_small_segmented_urg = seg.unseeded_segmentation(image_hela33_small_n, image_gt_hela33_small.copy(), (0, 0),
                                                          50, 0.01, 300)
    image_clipped_segmented_urg = seg.unseeded_segmentation(image_hela33_clipped, image_gt_hela33_small.copy(), (0, 0),
                                                            50, 0.1, 300)
    image_clipped_extreme_segmented_urg = seg.unseeded_segmentation(image_hela33_clipped_extreme,
                                                                    image_gt_hela33_small.copy(), (0, 0), 50, 0.1, 300)
    image_median_segmented_urg = seg.unseeded_segmentation(image_hela33_median, image_gt_hela33_small.copy(), (0, 0),
                                                           50, 0.1, 300)
    image_hela33_gauss = ip.gaussian_filter(image_hela33_small, 3)
    image_gauss_segmented_urg = seg.unseeded_segmentation(image_hela33_gauss, image_gt_hela33_small.copy(), (0, 0), 50,
                                                          0.01, 300)
    ip.show_six_images_colorbar(image_small_segmented_urg, image_clipped_segmented_urg,
                                image_clipped_extreme_segmented_urg, image_median_segmented_urg,
                                image_gauss_segmented_urg, image_gt_hela33_small, 0.32)


def results_nih3t3_seeded():
    """
    shows the result for seeded region growing on a complete image of the NIH3T3 data set
    :return: shows six images: the original one, the found regions, three differently  clipped ones and the ground truth
             image
    """
    image_intensity = sk.imread("Data/NIH3T3/img/dna-42.png")
    image_srg_dna_42 = sk.imread("Result_Pictures/Seeded_Region_Growing/NIH3T3/dna-42_srg_srg.tif")
    image_srg_dna_42_merged_clipped_undetected_cells = sk.imread(
        "Result_Pictures/Seeded_Region_Growing/NIH3T3/dna-42_merged_0.057_200_clipped_undetected_cells.tif")
    image_srg_dna_42_merged_clipped_background = sk.imread(
        "Result_Pictures/Seeded_Region_Growing/NIH3T3/dna-42_merged_0.056_200_clipped_background_merged.tif")
    image_srg_dna_42_merged_clipped_manually = sk.imread(
        "Result_Pictures/Seeded_Region_Growing/NIH3T3/dna-42_clipped_merged_manuel.tif")
    image_gt_dna_42 = sk.imread("Data/NIH3T3/gt/42.png")
    image_gt_dna_42 = ds.final_clipping(image_gt_dna_42)

    ip.show_three_images_colorbar(image_intensity, image_gt_dna_42, image_srg_dna_42, 0.35)
    ip.show_three_images_colorbar(image_srg_dna_42_merged_clipped_undetected_cells,
                                  image_srg_dna_42_merged_clipped_background, image_srg_dna_42_merged_clipped_manually,
                                  0.35)


def results_nih3t3_unseeded():
    """
    shows the result for unseeded region growing on a complete image of the NIH3T3 data set
    :return: shows two images: the image after merging and filtering and the image clipped after merging and filtering
    """
    image_urg_t01_merged_filtered = sk.imread(
        "Result_Pictures/Unseeded_Region_Growing/NIH3T3/dna-42_merging_0.07_10000_median_3.tif")
    image_urg_t01_merged_filtered_clipped = ds.final_clipping(image_urg_t01_merged_filtered)

    ip.show_two_images_colorbar(image_urg_t01_merged_filtered, image_urg_t01_merged_filtered_clipped, 0.54)
