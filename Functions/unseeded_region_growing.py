import numpy as np


# import skimage.io as sk
# import matplotlib.pyplot as plt
# import math as m
# from Functions import image_processing as ip


def unseeded_mean_region(img, reg):
    mean_value = []
    region_max = int(max(reg.flatten()))  # calculates amount of regions
    for count in range(1, region_max + 1):  # iterates over every region
        intensity = []
        for p in np.ndindex(img.shape):  # iterates over every pixel in the image
            if reg[p] == count:
                intensity.append(img[p])  # appends intensity value, if it is in the region
        mean_value.append(np.mean(intensity))  # calculates mean value of region
    return mean_value  # returns list with average of every region


def unseeded_add_neighbors(img, p):  # p describes pixel for which neighbors need to be added
    neighbors = []
    if p[0] > 0:  # Add neighbours to list T, up
        a = (p[0] - 1, p[1])
        neighbors.append(a)
    if p[0] < img.shape[0] - 1:  # Add neighbours to list T, down
        b = (p[0] + 1, p[1])
        neighbors.append(b)
    if p[1] > 0:  # Add neighbours to list T, left
        c = (p[0], p[1] - 1)
        neighbors.append(c)
    if p[1] < img.shape[1] - 1:  # Add neighbours to list T, right
        d = (p[0], p[1] + 1)
        neighbors.append(d)
    return neighbors


def unseeded_distance(img, neighbors, reg):  # img intensity values, regions is region number, list of neighbours
    means = unseeded_mean_region(img, reg)
    result = np.full(img.shape, 500)  # Wert 500 damit berechnete Distanzen immer kleiner sind
    for i in neighbors:
        nei = unseeded_add_neighbors(img, i)  # Nachbarn des Pixels
        distance = []
        for j in nei:
            if reg[j] != 0:  # Nachbarn mit zugeordneter Region
                distance.append(abs(img[i] - means[int(reg[j] - 1)]))  # Distanz Berechnung
        result[i] = min(distance)  # Minimale Distanz wird in den array geschrieben
    return result


def unseeded_pixel_pick(dis):  # distance ist Ergebnis von unseeded_distance, array mit Distanzen
    minimum = np.amin(dis)  # Minimale Distanz
    for p in np.ndindex(dis.shape):  # alle Pixel checken
        if dis[p] == minimum:  # Wenn Distanz minimaler Distanz entspricht
            pick = p  # Liste an Pixeln mit minimaler Distanz
            break
    return pick


# Regions Zuweisung z
# Distanz nochmal berechnen, zu Nachbarn mit region, und der gleichen Region zuweisen
def unseeded_region_direct(img, reg, pick):  # pick ist ausgesuchter Pixel
    means = unseeded_mean_region(img, reg)
    nei = unseeded_add_neighbors(img, pick)  # Nachbarn des Pixels
    dis = []
    region_number = []
    for j in nei:  # Distanz für Nachbarn wird wieder bestimmt
        if reg[j] != 0:  # Nachbarn mit zugeordneter Region
            dis.append(abs(img[pick] - means[int(reg[j] - 1)]))  # Distanz Berechnung
            region_number.append(reg[j])
    minimum = min(dis)  # Minimale Distanz
    for d in range(0, len(dis)):
        if dis[d] == minimum:  # Bestimmt Region mit niedrigster Distanz
            reg[pick] = region_number[d]  # Region vom Pixel mit kleinster Distanz wird übernommen
    return reg


# Zu nächst ähnlicher Region zuweisen
# Distanz zu allen Regionen ausrechnen und zu der mit der kleinsten Distanz zuweisen
def unseeded_region_indirect_or_new(img, reg, pick, t):
    means = unseeded_mean_region(img, reg)  # Mittelwerte Regionen
    dis = []
    for m in means:
        dis.append(abs(img[pick] - m))  # Alle Abstände zu den Regionen berechnen
    minimum = min(dis)  # Minimale Distanz
    if minimum < t:  # Distanz muss kleiner sein als der Threshold
        for d in range(0, len(dis)):  # Für alle Distanz Werte wir überprüft
            if dis[d] == minimum:  # Wenn die Distanz gleich dem Minimum ist
                reg[pick] = d + 1  # Letzte Region mit minimalem Wert
        else:  # Definition einer neuen Region, wenn keine passende gefunden wurde
            region_max = int(max(reg.flatten()))  # Aktuell höchste Regions-Nummer
            reg[pick] = region_max + 1  # Neuer Regions-Wert für Pixel pick
    return reg
