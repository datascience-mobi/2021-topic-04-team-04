import numpy as np
import skimage.io as sk
import matplotlib.pyplot as plt
from Functions import image_processing as ip
from Functions import seeded_region_growing as srg
from Functions import unseeded_region_growing as urg
from Functions import seed_detection as sd
from Functions import region_merging as rm
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
    dice_score = [0.786, 0.906, 0.972, 0.973, 0.965, 0.965, 0.785, 0.973, 0.756, 0.935]
    segmentation_method = ["Seeded", "Unseeded", "Seeded", "Unseeded", "Seeded", "Unseeded", "Seeded", "Unseeded",
                           "Seeded", "Unseeded"]
    preprocessing = ["unprocessed", "unprocessed", "clipped", 'clipped', 'extreme clipped', 'extreme clipped', 'median',
                     'median', 'gauss', 'gauss']

    df = pd.DataFrame(list(zip(dice_score, segmentation_method, preprocessing)), columns=columns_names)

    ax = sns.barplot(x="Preprocessing", y="Dice Score", hue="Segmentation Method", data=df, palette="dark")
    ax.set(ylim=(0.7, 1))
    plt.legend(loc=3)


def barplot_results():
    """
    creates a barplot for the dice scores of the segmented images of the different data sets
    :return: barplot for the different data sets
    """
    columns_names = ["Dice Score", "Segmentation Method", "Data Sets"]
    dice_score = [0.865, 0.868, 0.767, 0.861, 0.44, 0.55]
    segmentation_method = ["Seeded", "Unseeded", "Seeded", "Unseeded", "Seeded", "Unseeded"]
    data_sets = ["N2DH-GOWT1", "N2DH-GOWT1", "N2DL-HeLa", "N2DL-HeLa", "NIH3T3", "NIH3T3"]

    df = pd.DataFrame(list(zip(dice_score, segmentation_method, data_sets)), columns=columns_names)

    ax = sns.barplot(x="Data Sets", y="Dice Score", hue="Segmentation Method", data=df, palette="dark")
    ax.set(ylim=(0.40, 1))
    plt.legend(loc=3)


def barplot_runtime():
    columns_names = ["Runtime in ms", "Sd_runtime in ms", "Segmentation Method", "Algorithm version"]
    runtime = [9790, 24900, 678, 933]
    sd_time = [2500, 2120, 22.5, 36.5]
    segmentation_method = ["Seeded", "Unseeded", "Seeded", "Unseeded"]
    version = ["Old", "Old", "New", "New"]

    df = pd.DataFrame(list(zip(runtime, sd_time, segmentation_method, version)), columns=columns_names)
    #  print(df)

    ax = sns.barplot(x="Algorithm version", y="Runtime in ms", hue="Segmentation Method", data=df, palette="dark")
    ax.set(ylim=(0, 27000))
    plt.legend(loc=3)


def load_images():
    """
    loads example images of all the data sets
    :return: graph with three images, one of each data set
    """
    image_intensity = sk.imread("Data/N2DH-GOWT1/img/t01.tif")
    image_intensity_data2 = sk.imread("Data/N2DL-HeLa/img/t52.tif")
    image_intensity_data3 = sk.imread("Data/NIH3T3/img/dna-42.png")
    ip.show_three_images_title(image_intensity, image_intensity_data2, image_intensity_data3,
                               "Example images of different data sets", "N2DH-GOWT", "N2DL-Hela", "NIH3T3", 0.45, 0.75)


def clipping_examples():
    """
    clips images using different methods and shows the results in a graph
    :return: graph with the original image, the clipped one and the extremly clipped one
    """
    image_intensity = sk.imread("Data/N2DH-GOWT1/img/t01.tif")
    img_clipped = ip.image_clipping(image_intensity, 5, 25)
    img_clipped_extreme = ip.image_clipping_extreme(image_intensity, 5, 15)
    ip.show_three_images_title(image_intensity, img_clipped, img_clipped_extreme,
                               "Application of different clipping methods", "unprocessed", "clipped",
                               "clipped to border values", 0.45, 0.75)


def bright_spots_example():
    """
    removed bright spots using different methods and shows the results
    :return: graph with the original image, removed bright spot image and removed bright spot with border image
    """
    img_with_bright_spots = sk.imread("Data/NIH3T3/img/dna-33.png")
    img_removed_spots = ip.remove_bright_spots(img_with_bright_spots, 200, 60)
    img_removed_spots_with_border = ip.remove_bright_spots_with_border(img_with_bright_spots, 200, 60, 40)
    ip.show_three_images_title(img_with_bright_spots, img_removed_spots, img_removed_spots_with_border, "Clipping of "
                                                                                                        "bright "
                                                                                                        "spots",
                               "image with bright spots", "removed bright spots", "removed bright spots with border",
                               0.35, 0.7)


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
    image_regions_from_seeds = sd.reduce_region_number(image_regions_from_seeds, 7)

    image_regions = srg.region_growing(image_intensity_small, image_regions_from_seeds.copy())
    image_result_unseeded = urg.unseeded_region_growing_algorithm(image_intensity_small, (0, 0), 5)

    results_region_merging_similarity = rm.distance_merging_while(image_regions.copy(), 0.05, image_intensity_small)
    image_rm_similarity, inter_region_neighbors, means = results_region_merging_similarity
    image_rm_size = rm.region_merging_size(image_intensity_small, image_rm_similarity.copy(), inter_region_neighbors,
                                           means, 500)
    image_clipped_s = ds.final_clipping(image_rm_size.copy())
    ip.show_four_images_title(image_regions, image_rm_similarity, image_rm_size, image_clipped_s,
                              "Visualization of region merging after seeded region growing", "after region growing",
                              "region merging by similarity", "region merging by size", "final result", 0.335)

    results_region_merging_similarity_urg = rm.distance_merging_while(image_result_unseeded.copy(), 0.08,
                                                                      image_intensity_small)
    image_rm_similarity_urg, inter_region_neighbors_urg, means_urg = results_region_merging_similarity_urg
    image_rm_size_urg = rm.region_merging_size(image_intensity_small, image_rm_similarity_urg.copy(),
                                               inter_region_neighbors_urg, means_urg, 1000)
    image_clipped_s_urg = ds.final_clipping(image_rm_size_urg.copy())

    ip.show_four_images_title(image_result_unseeded, image_rm_similarity_urg, image_rm_size_urg, image_clipped_s_urg,
                              "Visualization of region merging after unseeded region growing", "after region growing",
                              "region merging by similarity", "region merging by size", "final result", 0.335)


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

    ip.show_four_images_title(image_intensity, image_srg_t01, image_srg_t01_merged_clipped, image_gt_t01,
                              "Result of image t01 of N2DH-GOWT1 after seeded region growing", "normal image",
                              "after seeded region growing", "final image", "ground truth image", 0.34)


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

    ip.show_four_images_title(image_urg_t01_merged, image_urg_t01_merged_clipped, image_urg_t01_merged_filtered,
                              image_urg_t01_merged_filtered_clipped,
                              "Effects of median filtering after region merging on image t01 of N2DH-GOWT1 after"
                              " unseeded region growing",
                              "merged image",
                              "final image after merging", "merged and filtered",
                              "final image after merging and filtering", 0.34)


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
    ip.show_six_images_two_rows_title(image, image_srg, image_srg_clipped, image_urg, image_urg_filtered_clipped, gt,
                                      "Results of seeded and unseeded region growing on image t13 of N2DL-Hela",
                                      "normal image", "seeded region growing", "final result srg",
                                      "unseeded region growing", "final result urg", "ground truth image", 0.85)
    dice_value_srg = ds.dice_score(image_srg_clipped, gt)
    print("Seeded Dice score: " + str(dice_value_srg))
    dice_value_urg = ds.dice_score(image_urg_filtered_clipped, gt)
    print("Unseeded Dice score: " + str(dice_value_urg))


def nih3t3_show_bright_spots():
    """
    shows the removal of a bright spot on a smaller picture of the NIH3T3 data set
    :return: three images are shown, the original one, one with normally removed bright spot and one with removed
    bright spot with border
    """
    image_intensity_d3_small = sk.imread("Data/NIH3T3/img/dna-42.png")[730:850, 1200:1300]
    img_removed_spots_intact_nuclei = ip.remove_bright_spots(image_intensity_d3_small, 130, 60)
    img_removed_spots_with_border = ip.remove_bright_spots_with_border(image_intensity_d3_small, 130, 60, 30)
    ip.show_three_images_title(image_intensity_d3_small, img_removed_spots_intact_nuclei,
                               img_removed_spots_with_border,
                               "Preprocessing of dna-42 image fragment (NIH3T3) to remove bright spots",
                               "image fragment", "clipping of bright spots", "clipping with border removal", 0.55, 0.8)


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

    result_seg_with_bright_spots, dice_score_with_bright_spots = seg.seeded_segmentation(image_intensity_d3_small,
                                                                                         gt_data3_small, 0.05, 0.1, 400)
    result_seg_removed_spots_intact_nuclei, dice_score_removed_spots = seg.seeded_segmentation(
        img_removed_spots_intact_nuclei, gt_data3_small,
        0.05, 0.26, 400)
    result_seg_removed_spots_with_border, dice_score_removed_spots_border = seg.seeded_segmentation(
        img_removed_spots_with_border, gt_data3_small, 0.05,
        0.2, 400)
    ip.show_three_images_title(result_seg_with_bright_spots, result_seg_removed_spots_intact_nuclei,
                               result_seg_removed_spots_with_border,
                               "Effects of bright spot removal on a dna-42 image fragment after seeded region growing",
                               "result of unprocessed image", "bright spot removal", "bright spot removal with border",
                               0.55, 0.8)
    print("Dice score with bright spots: " + str(dice_score_with_bright_spots))
    print("Dice score after removing bright spots: " + str(dice_score_removed_spots))
    print("Dice score after removing bright spots and border: " + str(dice_score_removed_spots_border))


def nih3t3_blurs():
    """
    shows the impact of blurs in the NIH3T3 data set on seeded region growing
    :return: shows three images, the original one, the found regions and the clipped one
    """
    image_dna33_small = sk.imread("Data/NIH3T3/img/dna-33.png")[200:500, 900:1200]
    srg_image_dna33_small = sk.imread("Result_Pictures/Unseeded_Region_Growing/NIH3T3/dna-33_5_blurs.tif")
    merged_image_small = rm.region_merging(srg_image_dna33_small.copy(), image_dna33_small, 0.01, 17500)
    clipped_image_small = ds.final_clipping(merged_image_small)
    ip.show_three_images_title(image_dna33_small, srg_image_dna33_small, clipped_image_small,
                               "Impact of blurs on seeded region growing of dna-33 (NIH3T3)", "dna-33 image fragment",
                               "seeded region growing", "final result", 0.45, 0.75)


def show_preprocessing():
    """
    shows the different preprocessing methods performed on a smaller image of the N2DL-HeLa data set and their impact
    in seeded and unseeded region growing
    :return: shows three sets of six images
    first set: image normalized, clipped, clipped extreme, median, gauss and ground truth
    second set: seeded region growing on the images of the first set and the ground truth image
    thirs set: unseeded region growing on the images of the first set and the ground truth image
    """
    image_hela13_small = sk.imread("Data/N2DL-HeLa/img/t13.tif")[100:200, 450:550]
    image_hela13_small_n = ip.subtract_minimum(image_hela13_small.copy())
    image_gt_hela13_small = sk.imread("Data/N2DL-HeLa/gt/man_seg13.tif")[100:200, 450:550]

    image_hela13_clipped = ip.image_clipping(image_hela13_small_n, 0.03 * np.amax(image_hela13_small_n),
                                             0.1 * np.amax(image_hela13_small_n))
    image_hela13_clipped_extreme = ip.image_clipping_extreme(image_hela13_small_n, 0.03 * np.amax(image_hela13_small_n),
                                                             0.1 * np.amax(image_hela13_small_n))
    image_hela13_median = ip.median_filter(image_hela13_small_n, 3)
    image_hela13_gauss = ip.gaussian_filter(image_hela13_small_n, 1)
    ip.show_six_images_title(image_hela13_small_n, image_hela13_clipped, image_hela13_clipped_extreme,
                             image_hela13_median, image_hela13_gauss, image_gt_hela13_small,
                             "Application of different preprocessing methods on image t13 fragment (N2DL-Hela)",
                             "unprocessed image", "clipped image", "extremely clipped image", "median filter sigma 3",
                             "gaussian filter sigma 1", "ground truth image", 0.3)

    image_small_segmented, dice_seeded_normal = seg.seeded_segmentation(image_hela13_small_n, image_gt_hela13_small,
                                                                        0.5, 0.1, 300)
    image_clipped_segmented, dice_seeded_clipped = seg.seeded_segmentation(image_hela13_clipped, image_gt_hela13_small,
                                                                           0.5, 0.1, 300)
    image_clipped_extreme_segmented, dice_seeded_clipped_extreme = seg.seeded_segmentation(image_hela13_clipped_extreme,
                                                                                           image_gt_hela13_small, 0.5,
                                                                                           0.1, 300)
    image_median_segmented, dice_seeded_median = seg.seeded_segmentation(image_hela13_median, image_gt_hela13_small,
                                                                         0.5, 0.1, 300)
    image_gauss_segmented, dice_seeded_gaussian = seg.seeded_segmentation(image_hela13_gauss, image_gt_hela13_small,
                                                                          0.5, 0.1, 300)
    ip.show_six_images_title(image_small_segmented, image_clipped_segmented, image_clipped_extreme_segmented,
                             image_median_segmented, image_gauss_segmented, image_gt_hela13_small,
                             "Results of differently preprocessed images after seeded region growing",
                             "unprocessed image", "clipped image", "extremely clipped image",
                             "median filter sigma 3",
                             "gaussian filter sigma 1", "ground truth image", 0.3)

    image_small_segmented_urg, dice_unseeded_normal = seg.unseeded_segmentation(image_hela13_small_n,
                                                                                image_gt_hela13_small.copy(), (0, 0),
                                                                                50, 0.01, 300)
    image_clipped_segmented_urg, dice_unseeded_clipped = seg.unseeded_segmentation(image_hela13_clipped,
                                                                                   image_gt_hela13_small.copy(), (0, 0),
                                                                                   50, 0.1, 300)
    image_clipped_extreme_segmented_urg, dice_unseeded_clipped_extreme = seg.unseeded_segmentation(
        image_hela13_clipped_extreme,
        image_gt_hela13_small.copy(), (0, 0), 50, 0.1, 300)
    image_median_segmented_urg, dice_unseeded_median = seg.unseeded_segmentation(image_hela13_median,
                                                                                 image_gt_hela13_small.copy(), (0, 0),
                                                                                 50, 0.1, 300)
    image_hela33_gauss = ip.gaussian_filter(image_hela13_small, 3)
    image_gauss_segmented_urg, dice_unseeded_gaussian = seg.unseeded_segmentation(image_hela33_gauss,
                                                                                  image_gt_hela13_small.copy(), (0, 0),
                                                                                  50,
                                                                                  0.01, 300)
    ip.show_six_images_title(image_small_segmented_urg, image_clipped_segmented_urg,
                             image_clipped_extreme_segmented_urg, image_median_segmented_urg,
                             image_gauss_segmented_urg, image_gt_hela13_small,
                             "Results of differently preprocessed images after unseeded region growing",
                             "unprocessed image", "clipped image", "extremely clipped image",
                             "median filter sigma 3",
                             "gaussian filter sigma 1", "ground truth image", 0.3)
    print("Dice score of normal image: Seeded region growing: " + str(
        dice_seeded_normal) + " Unseeded region growing:" + str(dice_unseeded_normal))
    print("Dice score of clipped image: Seeded region growing: " + str(
        dice_seeded_clipped) + " Unseeded region growing:" + str(dice_unseeded_clipped))
    print("Dice score of extremely clipped image: Seeded region growing: " + str(
        dice_seeded_clipped_extreme) + " Unseeded region growing:" + str(dice_unseeded_clipped_extreme))
    print("Dice score of median filtered image: Seeded region growing: " + str(
        dice_seeded_median) + " Unseeded region growing:" + str(dice_unseeded_median))
    print("Dice score of gaussian filtered image: Seeded region growing: " + str(
        dice_seeded_gaussian) + " Unseeded region growing:" + str(dice_unseeded_gaussian))


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

    ip.show_six_images_two_rows_title(image_intensity, image_gt_dna_42, image_srg_dna_42,
                                      image_srg_dna_42_merged_clipped_undetected_cells,
                                      image_srg_dna_42_merged_clipped_background,
                                      image_srg_dna_42_merged_clipped_manually,
                                      "Results of seeded region growing on dna-42 (NIH3T3) and evaluation of region "
                                      "merging",
                                      "normal image", "ground truth image", "after seeded region growing",
                                      "merging, several background regions", "merging, one background region",
                                      "manual merging of background", 0.85)


def results_nih3t3_unseeded():
    """
    shows the result for unseeded region growing on a complete image of the NIH3T3 data set
    :return: shows two images: the image after merging and filtering and the image clipped after merging and filtering
    """
    image_urg_t01_merged_filtered = sk.imread(
        "Result_Pictures/Unseeded_Region_Growing/NIH3T3/dna-42_merging_0.07_10000_median_3.tif")
    image_urg_t01_merged_filtered_clipped = ds.final_clipping(image_urg_t01_merged_filtered)

    ip.show_two_images_colorbar(image_urg_t01_merged_filtered, image_urg_t01_merged_filtered_clipped, 0.54)
