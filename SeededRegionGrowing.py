#%%

%pylab
%matplotlib inline

from Functions import functions as f
import skimage.io as sk
import matplotlib as mb
import pylab as py

img = sk.imread("Data/N2DH-GOWT1/img/t01.tif") #Bild laden

f.show_image(img, 15, 8)
