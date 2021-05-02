# %pylab
# %matplotlib inline
#import skimage.io as sk
#from matplotlib import pyplot as pp

#import numpy
#import matplotlib
#from matplotlib import pylab, mlab, pyplot
#np = numpy
#plt = pyplot

#from IPython.display import display
#from IPython.core.pylabtools import figsize, getfigs

#from pylab import *
#from numpy import *

import matplotlib.pyplot as plt
import numpy as np



def show_image(img, x, y): # Funktion um Bilder schneller schön anzuzeigen
    plt.figure(figsize=(x,y))
    plt.imshow(img, "gray")
    plt.colorbar()

def sum(x,y):
    return x+y
