import numpy as np
from Functions import seeded_region_growing as srg

# import skimage.io as sk
# import matplotlib.pyplot as plt
# import math as m
# from Functions import image_processing as ip


def unseeded_distance(img, neighbors, reg):
    """
    :param img: Benutztes Bild
    :param neighbors: Liste an Nachbarn der Regionen
    :param reg: Array mit Regionen
    :return: Array mit Distanzen, für alle Pixel die keine Nachbarn sind ist 500 eingetragen
    """  # img intensity values, regions is region number, list of neighbours
    means = srg.mean_region(img, reg)
    dis = np.full(img.shape, 500)  # Wert 500 damit berechnete Distanzen immer kleiner sind
    nearest_region = np.zeros(img.shape)
    for i in neighbors:
        nei = srg.add_neighbors(img, i)  # Nachbarn des Pixels
        distance_list = []
        region_number = []
        for j in nei:
            if reg[j] != 0:  # Nachbarn mit zugeordneter Region
                distance_list.append(abs(img[i] - means[int(reg[j] - 1)]))  # Distanz Berechnung
                region_number.append(reg[j])
        dis[i] = min(distance_list)  # Minimale Distanz wird in den array geschrieben
        nearest_region[i] = region_number[distance_list.index(min(distance_list))]
    return dis, nearest_region


def unseeded_pixel_pick(dis):
    """
    :param dis: Array mit Distanzen
    :return: Position des Pixels mit der minimalen Distanz
    """  # distance ist Ergebnis von unseeded_distance, array mit Distanzen
    x = np.where(dis == np.amin(dis))  # Minimale Distanz
    minimum = list(zip(x[0], x[1]))[0]
    pick = (int(minimum[0]), int(minimum[1]))
    return pick


def unseeded_region_direct(reg, pick, nearest_region):
    """
    :param nearest_region: Array mit der Nummer der Region die die kleinste Distanz aufweist
    :param reg: Array mit Regionen
    :param pick: Ausgewählter Pixel
    :return: Array mit aktualisierten Regionen
    """
    reg[pick] = nearest_region[pick]  # Region vom Pixel mit kleinster Distanz wird übernommen
    return reg


def unseeded_region_indirect_or_new(img, reg, pick, t):
    """ # Zu nächst ähnlicher Region zuweisen
        # Distanz zu allen Regionen ausrechnen und zu der mit der kleinsten Distanz zuweisen
    :param img: Benutztes Bild
    :param reg: Array mit Regionen
    :param pick: Ausgewählter Pixel
    :param t: Threshold um zu entscheiden ob Pixel neue Region bilden soll
    :return: Array mit aktualisierten Regionen
    """
    means = srg.mean_region(img, reg)  # Mittelwerte Regionen
    distance_list = []
    for m in means:
        distance_list.append(abs(img[pick] - m))  # Alle Abstände zu den Regionen berechnen
    minimum = min(distance_list)  # Minimale Distanz
    if minimum < t:  # Distanz muss kleiner sein als der Threshold
        reg[pick] = distance_list.index(minimum) + 1  # Erste Region mit minimalem Wert
    else:  # Definition einer neuen Region, wenn keine passende gefunden wurde
        region_max = int(max(reg.flatten()))  # Aktuell höchste Regions-Nummer
        reg[pick] = region_max + 1  # Neuer Regions-Wert für Pixel pick
    return reg
