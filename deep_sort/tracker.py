from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    # metric permet d'avoir une mesure de distance
    # max_age = nombre maximum d'echecs avec la supression d'un trackage
    # n_init est le nombre de détections avant le trackage
    # filtre Kalaman afin de suivre les trajectoires
    # track est l'ensemble des éléments qui sont trackés


    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3): # constructeur qui permet d'enregistrer les différente valeurs
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = [] # création de l'ensemble d'un vecteur pour enregistrer tout les trackages
        self._next_id = 1

    def predict(self): #cette fonction est appelée avant chaque mise à jour du tracking

        for track in self.tracks: #pour tout les éléments qui sont en train d'être trackés
            track.predict(self.kf) # application de la prédiction sur chaque élément

    def update(self, detections):

        # cette fonction permet de mettre à jour les mesures et la gestion des trackages
        # detection est la liste d'objet de la classe detection

        # permet d'obtenir les tracks et detéctions associées , les tracks non associés et les détections non associées
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # mise à jour de l'ensemble des tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx]) # mise à jour des tracks et des détections dans les éléments associés
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed() #regarde si le tracking est raté
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx]) #création de nouveau tracking
        self.tracks = [t for t in self.tracks if not t.is_deleted()] # garde les tracks qui ne sont pas supprimés

        # mise à jour de la distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()] #enregistrement des trackings confirmé
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed(): #arrêt losque le trackage n'est pas dans l'état confirmé
                continue
            features += track.features # ajout de l'éléments features au vecteur
            targets += [track.track_id for _ in track.features] # ajout des id des tracks au vecteur
            track.features = [] # remise à zero
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets) #utilisation de la fonction de nn_matching pour la mise a jour de la distance metric avec les nouvelles données

    def _match(self, detections):
        # cette fonction permet d'associer les detections et les tracks qui vont enssemble

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices]) # céation d'un vecteur composé des éléments feature des détections
            targets = np.array([tracks[i].track_id for i in track_indices]) #création d'un vecteur avec l'indice de chaque track
            cost_matrix = self.metric.distance(features, targets) # calcul de la cost_matrix
            cost_matrix = linear_assignment.gate_cost_matrix(self.kf, cost_matrix, tracks, dets, track_indices,detection_indices) #modification de la cost_matrix

            return cost_matrix

        # Diviser l'ensemble des tracks en tracks confirmées et non confirmées.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()] # enregistrement  des objets où le tracking est confirmé
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()] # enregistrement  des objets où le tracking est non-confirmé

        # associe les pistes confirmées, un vecteur des tracks et détections associées , un vecteur des tracks non associés et un vecteur de detections non associées
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(gated_metric, self.metric.matching_threshold, self.max_age,self.tracks, detections, confirmed_tracks)

        # Associe les tracks restants avec les tracks non confirmés en utilisant IOU
        iou_track_candidates = unconfirmed_tracks + [ k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1] # prend les candidats possible pour les nouvelles associations
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1] # enregistrement les tracks qui ne sont pas dans les nouveaux candidats
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching( iou_matching.iou_cost, self.max_iou_distance, self.tracks,detections, iou_track_candidates, unmatched_detections)
        # utilisation de la fonction min_cost_matching de linear_assignment pour calculer les nouveaux éléments associés et non associés

        matches = matches_a + matches_b # réuni les 2 vecteurs qui possédent les éléments associés
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b)) # réuni les 2 vecteurs qui possédent les tracks non associés
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah()) #permet d'obtenir le vecteur de coordonées de la box avec la même dimension que la matrice de covariance
        self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age,detection.feature))
        #ajout au vecteur track l'élément de la class track avec ses différentes données
        self._next_id += 1 # augmente la valeur de l'id pour la prochaine détection