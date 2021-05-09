from __future__ import absolute_import
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from . import kalman_filter


INFTY_COST = 1e+5


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):

    # distance_metric: Callable [List [Track], List [Detection], List [int], List [int])
    # La métrique de distance reçoit une liste de tracks et de détections ainsi que
    # une liste de N indices de tracks et de M indices de détection.
    # renvoit une cost_matrice de dimension NxM, où l'élément (i, j) est le
    # coût d'association entre la i-ème tack dans les indices de track donnés et
    # la j-ème détection dans les indices de détection donnés.

    # max_distance : =0.5 Les associations dont le coût est supérieur à cette valeur sont ignorés.
    # cascade_depth (float) :  nombre maximum d'échecs avant la supression d'un trackage
    # tracks : liste des éléments trackés
    # detections : liste des éléments détectés
    # track_indices : liste des indices qui mappent les lignes de cost_matrix avec les élements de track
    # detection_indices : liste des indices  qui mappent les colonnes de cost_matrix avec les élements de détection

    # return :  (List[(int, int)], List[int], List[int]) qui est composé:
    # une liste de tracks et de detections qui correspondent
    # une liste d'indices de tracks qui ne correpondent pas
    # une liste d'indices de detection qui ne correpondent pas
    if track_indices is None:
        track_indices = np.arange(len(tracks)) # création du vecteur avec les indices de tracks
    if detection_indices is None:
        detection_indices = np.arange(len(detections)) # création du vecteur avec les indices de détections

    if len(detection_indices) == 0 or len(track_indices) == 0: # si un des 2 vaut 0 impossible de les faire correspondre
        return [], track_indices, detection_indices  # impossible d'associer.

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices) # appelle de la fonction distance_metric de tracker pour calculer la cost_metric
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5 #lorsque la valeur est au dessus de la valeur maximum
    indices = linear_assignment(cost_matrix) # renvoi les paires d'indices (ligne, colonne)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices): #col permet d'avoir la position et detection_idx la valeur des éléments de détections_indices
        if col not in indices[:, 1]:   # si l'indice ne se trouve pas dans la deuxième  colonne d'indices
            unmatched_detections.append(detection_idx)   #ajout dans le vecteur des détections non associées
    for row, track_idx in enumerate(track_indices): #col permet d'avoir la position et détection_idx la valeur des éléments de track_indices
        if row not in indices[:, 0]:  # si l'indice ne se trouve pas dans la première colonne d'indices
            unmatched_tracks.append(track_idx)    #ajout dans le vecteur des tracks non associés
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:  #losrque la distance est trop grande on ajoute dans les vecteurs non associés
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx)) #ajout des tracks et détections associées
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):

    #distance_metric: Callable [List [Track], List [Detection], List [int], List [int])
    #La métrique de distance reçoit une liste de tracks et de détections ainsi que
    #une liste de N indices de tracks et de M indices de détection. La métrique doit
    #renvoit une cost_matrice de dimension NxM, où l'élément (i, j) est le
    #coût d'association entre la i-ème tack dans les indices de track donnés et
    #la j-ème détection dans les indices de détections données.

    #max_distance : =0.5 Les associations dont le coût est supérieur à cette valeur sont ignorés.
    #cascade_depth (float) :  nombre maximum d'echecs avec la supression d'un trackage
    # tracks : liste des éléments trackés
    # detections : liste des éléments détectés
    # track_indices : liste des indices des éléments trackés
    # detection_indices : les des indices des éléments détecté

    #return :  (List[(int, int)], List[int], List[int]) qui est composé:
    # une liste de tracks et de detections qui se correspondent
    # une liste d'indices de tracks qui ne correpondent pas
    # une liste d'indices de detection qui ne correpondent pas

    if track_indices is None:
        track_indices = list(range(len(tracks))) # création d'une liste avec les indices des tracks
    if detection_indices is None:
        detection_indices = list(range(len(detections))) # création d'une liste avec les indices des détections

    unmatched_detections = detection_indices # enregistre la liste
    matches = []
    for level in range(cascade_depth): # pour le nombre maximum d'échecs on essaye d'associer les éléments trackés et les éléments détectés
        if len(unmatched_detections) == 0:  # si pas de detections break
            break

        track_indices_l = [k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ] # enregistre l'indice de l'élément tracké si  nombre total d'images aprés la derniére mise à jour est bien égale au moment où l'on se trouve
        if len(track_indices_l) == 0:  # rien à égaler
            continue

        matches_l, _, unmatched_detections = \
            min_cost_matching(distance_metric, max_distance, tracks, detections,track_indices_l, unmatched_detections) #permet d'avoir les vecteurs des éléments associé , des track et des detections non associé
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches)) #enléve les éléments de track associé
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, detection_indices,gated_cost=INFTY_COST, only_position=False):

    # but de la fonction est d'invalider les entrées de la matrice de perte (cost matrix) en fonction de l'état distributions obtenues par filtrage de Kalman.
    # permet de renvoyer la matrice de coût modifié

    #kf = kalman filter

    #cost_matrix: La matrice de coût dimensionnel NxM, où N est le nombre d'indices de trackages et M est le nombre d'indices de détection, de sorte que l'entrée (i, j) est le
    #coût d'association entre `trackages [track_indices [i]]` et`détections [détection_indices [j]]`.
    # track : liste des trackages prédisent au moment actuel
    #Detections : liste des détections
    #track_indices: Liste des indices de track qui mappent les lignes de `cost_matrix` avec les tracks de trackage
    #detection_indices : liste des incides qui mappent les colonnes de cost_matrix aux detections

    # return la cost matrix modifié



    gating_dim = 2 if only_position else 4 #prends comme valeur 4 par défaut
    gating_threshold = kalman_filter.chi2inv95[gating_dim] #prends la valeur de l'élément 4 de chi2inv95
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices]) #enregistre les différentes valeurs (x,y,(w/h),h) de chaque détection
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx] #enregistre l'élément tracker
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position) #permet de calculer les distances
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost # prend la valeur de INFTY_COST si les distances sont au dessus du seuil
    return cost_matrix