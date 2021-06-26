# Report

---

### Introduction

Image segmentation is an image processing program in the field of data analysis, whose role has become increasingly important in recent years. It consists in grouping pixels of the image by commonalities of image intensities and thus creating regions of pixels that can be used for data analysis and evaluation. In this way, image content can be simplified, and its representation can be modified by using methods like thresholding, boundary or region-based segmentation or hybrid techniques. Image segmentation is often the first step in image analysis. It is widely used, especially in the medical field, but also in the processing of geographical data.
Region growing is a subset of the hybrid techniques of image segmentation which can be used for cell nuclei segmentation. 
This class of methods can be further subdivided into seeded and unseeded region growing, while the focus in this project lies on seeded region growing. Here, selected initial seeds are assigned pixels from the neighborhood if their relative Euclidean distance to the seeds is below a certain threshold. Subsequently, certain regions are merged to combine similar or small regions.
After implementing the seeded region growing algorithm, its performance will be evaluated. For this, result images are compared to already segmented gt-images by using the Dice score. Finally, seeded and unseeded region growing are also compared with each other using the Dice Score.

---

### Datasets – N2DH-GOWT1, N2DL-HeLa and NIH3T3

In order to run and test the implemented seeded region growing algorithm, three different datasets were used: N2DH-GOWT1, N2DL-HeLa and NIH3T3. The datasets consist of a total of 28 image files, in which different structures of the nucleus were stained using the fluorescent protein GFP. Attached to the data sets are gt-images in which the nuclei have already been segmented.
Each dataset has specific challenges that need to be considered. 
In the first dataset N2DH-GOWT1, transcription factor Oct 4 was stained by GFP in mouse embryonic stem cells. The cells are well in focus and do not overlap. There are no reflections or blurs. However, the stained cells differ greatly in brightness. Compared to the other data sets, N2DH-GOWT1 is considered to be the easiest to work with.
N2DL-HeLa shows GFP labeled core histone 2b proteins from human cervical carcinomas. Overall, the cell density is high whereas the resolution is low, and the cells vary greatly in brightness. There are no interfering effects due to reflections or blurs.
The last dataset NIH3T3 consists of mouse embryo fibroblasts. EGFP was used to label the CD-tagged protein. In NIH3T3 the cells are displayed in a very low contrast and differ strongly in their size and shape. Very bright regions, presumably caused by reflections from the microscope, probably make this data set the most challenging.
### Ursprung der Datensätze muss noch hinzugefügt werden.
