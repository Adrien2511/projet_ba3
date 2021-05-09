class TrackState:


    #Enumeration des piste pour le trackage provisoire, quand il y a suffisament d'élément elles sont dans état confirmées
    # et quand elles ne sont plus confirmées elles sont dans l'état supprimées

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:

    # mean : est un vecteur de l'état initial [(x,y) au centre, largeur/hauteur , hauteur ]
    # covariance : matrice de covariance du vecteur mean
    # track_id : id de chaque détection
    # n_init : nombre de détections avant que le trackage soit confirmé
    # max_age : nombre maximum d'échecs avant de passer dans l'état supprimé
    # hits : nombre total d'updates
    # age : nombre total d'images depuis le début
    # time_since_update : nombre total d'images aprés la derniére mise à jour
    # state : état actuel
    # features : Vecteur de caractéristiques de la détection d'où provient cette piste.
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None): #constructeur de la classe
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self): #permet de transformer les coordonées de la box en : haut gauche (x,y) , largeur hauteur

        ret = self.mean[:4].copy() #création d'une copie de mean des indices de 0 à 3

        ret[2] *= ret[3]  #multiplie l'élément d'indice 2 par celui d'indice 3
        ret[:2] -= ret[2:] / 2 #permet d'obtenir la position en haut à gauche en enlevant la moitié de la hauteur et de la largeur du point au centre
        return ret

    def to_tlbr(self): #permet de transformer les coordonées de la box en : (min x, min y, max x, max y)

        ret = self.to_tlwh() #appelle de la fonction to_tlwh
        ret[2:] = ret[:2] + ret[2:]  #permet d'obtenir les éléments en position max en addition la largeur et la hauteur
        return ret

    def predict(self, kf):
        #Propager la distribution de l'état au pas de temps actuel en utilisant l'étape de prédiction du filtre de Kalman
        #kf = Kalman filter

        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1 #augmente le nombre d'images utilisées
        self.time_since_update += 1 #augmenter le nombre d'images utilisées aprés la derniére mise à jour

    def update(self, kf, detection):

        #Effectuer l'étape de mise à jour des mesures du filtre de Kalman et des caractéristiques.
        #kf :kalman filter
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature) #ajout de l'élément detection

        self.hits += 1   #augmente le nombre d'uptade
        self.time_since_update = 0 #remise à 0 du nombre d'images utilisées aprés chaque mise à jour
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            #vérifie si l'état est en tentative de detection et si le nombre d'uptade est supérieur au nombre initial
            self.state = TrackState.Confirmed  #change l'état en confirmé

    def mark_missed(self):

        #marque l'objet si le trackage est raté
        if self.state == TrackState.Tentative: #vérifier si l'état est en tentavie de trackage
            self.state = TrackState.Deleted  #passe dans l'état supprimé si c'est vrai
        elif self.time_since_update > self._max_age: #regarde si le nombre total d'images est au dessus du seuil d'échec
            self.state = TrackState.Deleted #passe dans l'état supprimé si c'est vrai

    def is_tentative(self):
        #return true si l'objet à une possibilté dêtre traqué ==> state == 1
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        # return tue si l'objet est tracké ==> state == 2
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        #returne true si le trackage de l'objet est fini est qu'il doit être supprimé ==> state == 3
        return self.state == TrackState.Deleted