import numpy as np

# ce fichier permet de calculer les distances
# features représente les caractéristiques des images comprise dans les boxs
def _pdist(a, b):
    #Calculer la distance au carré par paire entre les points dans` a` et `b`.

     #a: Une matrice NxM de N échantillons de dimension M
     #b: Une matrice LxM de L échantillons de dimension M
     #Renvoi une matrice de taille len (a), len (b) telle que les élements (i, j)
     #contient la distance au carré entre `a [i]` et `b [j]`.

    a, b = np.asarray(a), np.asarray(b) # création de vecteur avec a et b
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b))) # si pas d'éléments dans a ou b alors renvoi une matrice de zeros
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):

    #Calculer la distance cosinus par paire entre les points dans` a` et `b
    # a: Une matrice NxM de N échantillons de dimension M
    # b: Une matrice LxM de L échantillons de dimensionn M
    # data_is_normalized: Si True, suppose que les lignes de a et b sont des vecteurs de longueur unitaire.
    # Sinon, a et b sont explicitement normalisés à la longueur 1
    # return : une matrice de taille len (a), len (b) telle que les élément (i, j)
    # contient la distance au carré entre `a [i]` et `b [j]`.

    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):

    # x : matrice de n vecteur colonnes (points d'échantillonnage)
    # y : matrice de M vecteur colonnes (points de requête)
    # return : un vecteur de taille M qui contient pour chaque entrée dans `y` le  plus petite distance euclidienne à un échantillon en «x».
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))



def _nn_cosine_distance(x, y):

    # aide à calculer la disance avec le voisin le plus proche
    # x : matrice de n vecteur colonnes (points d'échantillonnage)
    # y : matrice de M vecteur colonnes (points de requête)
    # return : un vecteur de taille M qui contient pour chaque entrée dans `y` le  plus petite distance cosinus à un échantillon en «x».

    distances = _cosine_distance(x, y) #utilisation de la fonction _cosinus_distance()
    return distances.min(axis=0) # prend le minimum de distance


class NearestNeighborDistanceMetric(object):

    # pour chaque cible, renvoi la distance la plus proche de tout échantillon observé jusqu'à présent
    # metric : prend comme valeur "euclidean" ou "cosinus"
    # matching_threshold :   Les échantillons avec une plus grande distance que ce seuil  sont considérés comme ayant une correspondance invalide
    # budget : S'il n'est pas nul, fixe les échantillons par classe à ce nombre au maximum.
    # sample : Un dictionnaire qui mappe des identités cibles à la liste d'échantillons qui ont été observées jusqu'à présent.

    def __init__(self, metric, matching_threshold, budget=None): #constructer

        # ce blox permet d'enregistrer la valeur de metric
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Métrique non valide ; doit être soit 'euclidean' or 'cosine'")


        self.matching_threshold = matching_threshold # enegistrement du seuil
        self.budget = budget
        self.samples = {} # création du dictionnaire

    def partial_fit(self, features, targets, active_targets):

        # mise a jour de la distance metric avec de nouvelles données
        # features : matrice de taille NxM de N features de dimension M
        # targets : un vecteur d'entier
        # active_targets : list de targets qui sont présent actuellement
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature) # ajoute au dictionnaire sample , la clé target avec comme valeur [feature]
            if self.budget is not None:   # si il y a budget
                self.samples[target] = self.samples[target][-self.budget:] #diminue de la valeur budget la valeur du dico
        self.samples = {k: self.samples[k] for k in active_targets} # met le dectionnaire dans l'ordre de la liste active_targets

    def distance(self, features, targets):


        # but : calculer la distance entre les features et targets
        # features : Une matrice NxM de N features de dimensionnalité M.
        # targets : une liste des indices des éléments trackés pour les faire correspondre aux features
        # return : une matrice de coût de forme len (targets), len (features), où
        # l'élément (i, j) contient la distance quadratique la plus proche entre
        # `targets [i]` et `features [j]`
        cost_matrix = np.zeros((len(targets), len(features))) # créations de la cost_matrix
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features) # enregistrement des colonnes
        return cost_matrix