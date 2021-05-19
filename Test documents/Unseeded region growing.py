# unseeded region growing

from Functions import functions as f
import skimage.io as sk

img = sk.imread("../Data/N2DH-GOWT1/img/t01.tif")
img_resize = f.img_resize(img, 500, 500)
f.show_image(img_resize, 15, 8)

# Start pixle is at position [0,0] -> A1


# Region A1 festlegen, Liste für alle vorhandenen Regionen erstellen (Listen Ai)

# Nachbarn bestimmen (Liste T)

# Liste T erstellen, die alle Pixel enthält die an gefundene Regionen grenzen

# Pixel x aus Liste T auswählen

# Liste N(x) sind die direkten Nachbarn von Pixel x

# Difference measure: Formel 1 Paper, Differenz Intensität Pixel x zu der Durchschnitts-Intensität der Region Ai
# Pixel x und Ai müssen aneinander angrenzen

# Differenz von allen Pixeln in Liste T zu allen deren angrenzenden Regionen berechnen
# Pixel mit kleinstem Wert wird zu Pixel z

# Differenz von z zu Ai wird mit einem treshold t (wie bestimmen) verglichen
# Wenn kleienr als t, z wird zu Region hinzugefügt

# Wenn größer, suchen wie die ähnlichste Region, egal ob angrenzend oder nicht
# Vergleich der minimalen Differenz mit treshold t, wenn kleiner wird z dieser Region zugewiesen

# Wenn größer, Pixel z wir eine neue Region

# Aktualisierung Listen Ai

# Nachbarn bestimmen, Liste T ...

# Verschnellerung: Nur die Differenzen neu berechnen der erweiterten Region
