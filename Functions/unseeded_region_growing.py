import numpy as np
from Functions import seeded_region_growing as srg

# import skimage.io as sk
# import matplotlib.pyplot as plt
# import math as m
# from Functions import image_processing as ip


def unseeded_distance(img, neighbors, reg):  # img intensity values, regions is region number, list of neighbours
    means = srg.mean_region(img, reg)
    result = np.full(img.shape, 500)  # Wert 500 damit berechnete Distanzen immer kleiner sind
    for i in neighbors:
        nei = srg.add_neighbors(img, i)  # Nachbarn des Pixels
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
    means = srg.mean_region(img, reg)
    nei = srg.add_neighbors(img, pick)  # Nachbarn des Pixels
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
    means = srg.mean_region(img, reg)  # Mittelwerte Regionen
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
