import numpy as np
import cv2


def non_max_suppression(boxes, max_bbox_overlap, scores=None): #permet d'enlever les boxs qui se chevauchent trop

    # boxes : tableau des boxes en format (x, y, largeur, hauteur)
    # max_bbox_overlap : # la superposition maximale des boxes
    # return : les indices des détections non supprimées par la superposition
    if len(boxes) == 0: # si il y a pas de boxes dans le vecteur
        return []

    boxes = boxes.astype(np.float) # transformations en float
    pick = []

    x1 = boxes[:, 0] # enregistrement de la coordonée x du point en haut à gauche des boxs
    y1 = boxes[:, 1] # enregistrement de la coordonée y du point en haut à gauche des boxs
    x2 = boxes[:, 2] + boxes[:, 0] # enregistrement de la coordonée x du point en bas à droite des boxs
    y2 = boxes[:, 3] + boxes[:, 1] # enregistrement de la coordonée y du point en bas à droite des boxs

    area = (x2 - x1 + 1) * (y2 - y1 + 1) # calcul de la surface
    if scores is not None:     # si il y a les scores
        idxs = np.argsort(scores) # # création d'un vecteur dont les élements sont l'ordre des éléments de scores
    else:
        idxs = np.argsort(y2)       # création d'un vecteur dont les élements sont l'odre des élémetents de y2

    while len(idxs) > 0:            # tant que la taille du vecteur est plus grand que 0
        last = len(idxs) - 1        # enregistrement le dernier indice des éléments du vecteur
        i = idxs[last]              # prends la valeur de cet indice
        pick.append(i)              # ajout de l'élément au vecteur pick

        xx1 = np.maximum(x1[i], x1[idxs[:last]])  # la plus grande valeur du point x en haut à gauche  entre le dernier élément et les autres
        yy1 = np.maximum(y1[i], y1[idxs[:last]])  # la plus grande valeur du point y en haut à gauche  entre le dernier élément et les autres
        xx2 = np.minimum(x2[i], x2[idxs[:last]])  # la plus grande valeur du point x en bas à droitr  entre le dernier élément et les autres
        yy2 = np.minimum(y2[i], y2[idxs[:last]])  # la plus grande valeur du point y en bas à droite  entre le dernier élément et les autres

        w = np.maximum(0, xx2 - xx1 + 1)         # la largeur de la box
        h = np.maximum(0, yy2 - yy1 + 1)         # la hauteur de la box

        overlap = (w * h) / area[idxs[:last]]    # calcul de la superposition

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > max_bbox_overlap)[0]))) #supprime l'élément si la supperposition est au dessus du seuil

    return pick