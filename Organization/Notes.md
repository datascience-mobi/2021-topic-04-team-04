## create conda environment

1. create file environment.yml in project folder of project you are working on
1. define name and dependencies of environment in environment.yml file: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#exporting-an-environment-file-across-platforms
1. commit
1. create and activate new environment https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
   1. conda env create -f environment.yml (execute on terminal)
   1. conda activate environmentGroup4
   1. conda env list (to test)
1. update environment: change environment.yml file, commit, update and activate https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#updating-an-environment
   1. conda env update --prefix ./env --file environment.yml  --prune
1. push, others can pull now environment.yml file and continue in the same way

## Create Python package
- https://www.tutorialsteacher.com/python/python-package 

## %pylab
- import numpy
- import matplotlib
- from matplotlib import pylab, mlab, pyplot
- np = numpy
- plt = pyplot

- from IPython.display import display
- from IPython.core.pylabtools import figsize, getfigs

- from pylab import *
- from numpy import *

## Ideas
- Mehrere Bilder in Liste einlesen (skimage.io.imread_collection)


## if __name__ == '__main__': 
- https://www.data-science-architect.de/__name____main__/

## dice score
def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice
- https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2

## use of image segmentation 
- https://imageannotation.home.blog/2020/06/18/what-is-the-application-of-image-segmentation/
- https://en.wikipedia.org/wiki/Image_segmentation

## code changes on 11.06.21
- Definitionen für Bildverarbeitung sind in image_processing und nicht mehr in functions
- Nur nötige Imports sind in den Functions files, der Rest ist auskommentiert darunter
- img wird in vielen Definitionen verwendet, soll im Code nicht verwendet werden,
  img in Funktionen und image beim Aufrufen
  reg in Funktionen und regions beim Aufrufen
  dis in Funktionen und distance beim Aufrufen
  pick in Funktionen und picked_pixel beim Aufrufen
  t in Funktionen und threshold beim Aufrufen
- Dokument functions ist gelöscht, die Funktionen sind in seed_detection

## Talk about:
- new_distance2 löschen?
- find_neighbors2 löschen?
- weitere Filter Iterationen auslagern?
- one_region_mean1 löschen?
- Standard_deviation_new ist ausgelagert
- warum funktioniert region_merging nur beim ersten Durchlauf

## ToDos:
- filtering auf zweitem Datensatz Gauß (Laura)
- kleine Punkte dna33 (Laura)
- dritter datensatz seeded bright spots weg (Marie), parameter
- Parameterliste (Laura)
- unseeded 3. datensatz dna42 (Ina)
- Text erster Datensatz vergleichen seeded unseeded (Marie)
- Text zweiter Datensatz Vergleich seeded unseeded (Ina und Jojo)
- Text dritter Datensatz Vergleich seeded unseeded (Ina)
- Diskussion (Marie)
- Abstract (Laura)
- PowerPoint Bilder (Jojo)
- PowerPoint Algorithmusüberblick (Marie), Überblick aller Code
- Diagramm Dice Scores (Marie)
- Algorythmusgegenüberstellung Unseeded, Seeded --> Text (Marie)