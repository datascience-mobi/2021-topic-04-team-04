import skimage.io as sk
import matplotlib.pyplot as plt
import numpy as np
import math as m
from Functions import image_processing as ip

def unseeded_find_neighbors(img, regions):
    Ne = []
    for p in np.ndindex(img.shape):
        if regions[p] != 0: # Pixels with region
            if p[0] > 0: # Add neighbours to list T, up
                a = (p[0]-1, p[1])
                if regions[a] == 0: # Nachbar darf noch keine Region haben
                    Ne.append(a)
            if p[0] < img.shape[0] - 1: # Add neighbours to list T, down
                b = (p[0]+1, p[1])
                if regions[b] == 0: # Nachbar darf noch keine Region haben
                    Ne.append(b)
            if p[1] > 0: # Add neighbours to list T, left
                c = (p[0], p[1]-1)
                if regions[c] == 0: # Nachbar darf noch keine Region haben
                    Ne.append(c)
            if p[1] < img.shape[1] - 1: # Add neighbours to list T, right
                d = (p[0], p[1]+1)
                if regions[d] == 0: # Nachbar darf noch keine Region haben
                    Ne.append(d)
    return Ne


def unseeded_mean_region(img, regions):
    mean_value = []
    region_max = int(max(regions.flatten()))  # calculates amount of regions
    for count in range(1, region_max + 1):  # iterates over every region
        intensity = []
        for p in np.ndindex(img.shape):
            if regions[p] == count:
                intensity.append(img[
                                     p])  # iterates over every pixel in the image and appends intensity value, if it is in the region
        mean_value.append(np.mean(intensity))  # calculates mean value of region
    return mean_value  # returns list with average of every region


def unseeded_add_neighbors(img, p): #p describes pixel for which neighbors need to be added
    Ne1 = []
    if p[0] > 0: # Add neighbours to list T, up
        a = (p[0]-1, p[1])
        Ne1.append(a)
    if p[0] < img.shape[0] - 1: # Add neighbours to list T, down
        b = (p[0]+1, p[1])
        Ne1.append(b)
    if p[1] > 0: # Add neighbours to list T, left
        c = (p[0], p[1]-1)
        Ne1.append(c)
    if p[1] < img.shape[1] - 1: # Add neighbours to list T, right
        d = (p[0], p[1]+1)
        Ne1.append(d)
    return Ne1


def unseeded_distance(img, list, regions): # img intensity values, regions is region number, list is list of neighbours
    means = unseeded_mean_region(img, regions)
    result = np.full(img.shape, 500) # Wert 500 damit berechnete Distanzen immer kleiner sind
    for i in list:
        nei = unseeded_add_neighbors(img, i) # Nachbarn des Pixels
        distance = []
        for j in nei:
            if regions[j] != 0: # Nachbarn mit zugeordneter Region
                distance.append(abs(img[i] - means[int(regions[j] - 1)]))# Distanz Berechnung
        result[i] = min(distance) # Minimale Distanz wird in den array geschrieben
    return result


def unseeded_pixle_pick(distance): # distance ist Ergebnis von unseeded_distance, array mit Distanzen
    min = np.amin(distance) # Minimale Distanz
    for p in np.ndindex(distance.shape): # alle Pixel checken
        if distance[p] == min: # Wenn Distanz minimaler Distanz entspricht
            pick = p # Liste an Pixeln mit minimaler Distanz
            break
    return pick


# Regions Zuweisung z
# Distanz nochmal berechnen, zu Nachbarn mit region, und der gleichen Region zuweisen
def unseeded_region_direct(img, regions, pick): # pick ist ausgesuchter Pixel
    means = unseeded_mean_region(img, regions)
    nei = unseeded_add_neighbors(img, pick) # Nachbarn des Pixels
    dis = []
    for j in nei: # Distanz für Nachbarn wird wieder bestimmt
        if regions[j] != 0: # Nachbarn mit zugeordneter Region
            dis.append(abs(img[pick] - means[int(regions[j] - 1)])) # Distanz Berechnung
        else: # Hohe Werte um die Reihenfolge der Liste zu erhalten
            dis.append(500)
    minimum = min(dis) # Minimale Distanz
    for d in range(0, len(dis)):
        if dis[d] == minimum: # Bestimmt Region mit niedrigster Distanz
            region_nei = nei[d] # nimmt den letzten Regionswert
            regions[pick] = regions[region_nei] # Regionswert vom nähsten Pixel wird übernommen
    return regions


# Zu nächst ähnlicher Region zuweisen
# Distanz zu allen Regionen ausrechnene und zu der mit der kleinsten Distanz zuweisen
def unseeded_region_indirect_or_new(img, regions, pick, t):
    means = unseeded_mean_region(img, regions)  # Mittelwerte Regionen
    dis = []
    for m in means:
        dis.append(abs(img[pick] - m)) # Alle Abstände zu den Regionen berechnen
    minimum = min(dis) # Minimale Distanz
    if minimum < t: # Distanz muss kleiner sein als der Threshold
        for d in range(0, len(dis)): # Für alle Distanz Werte wir überprüft
               if dis[d] == minimum: # Wenn die Distanz gleich dem Minimum ist
                    regions[pick] = d + 1 # Letzte Region mit minimalem Wert
        else:# Definition einer neuen Region, wenn keine passende gefunden wurde
           region_max = int(max(regions.flatten())) # Aktuell höchste Regionsnummer
           regions[pick] = region_max + 1 # Neuer Regionswert für Pixel pick
    return regions