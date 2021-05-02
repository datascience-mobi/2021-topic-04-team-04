# %pylab
# %matplotlib inline
import skimage.io as sk

def show_image(img, x, y): # Funktion um Bilder schneller schön anzuzeigen
    figure(figsize=(x,y))
    imshow(img, "gray")
    colorbar()

def sum(x,y):
    return x+y
