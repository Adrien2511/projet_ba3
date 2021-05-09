
import numpy as np
import scipy.linalg


"""
Tableau pour le quantile 0,95 de la distribution du chi carré avec N degrés de
liberté (contient des valeurs pour N = 1, ..., 9). Tiré du chi2inv de MATLAB / Octave
fonction et utilisé comme seuil de déclenchement Mahalanobis.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):

    #Un filtre de Kalman permet de  suivre les boxes sur image.
    #En utilisant  8 dimensions x, y, a, h, vx, vy, va, vh
    #centre du cadre (x, y)
    # a = largeur/hauteur
    # h = la hauteur h,
    # et leurs vitesses respectives.
    # la vitesse du modèle est considérée comme constante.


    def __init__(self):     #constructeur de la classe
        ndim, dt = 4, 1.


        # création des matrices du modèle Kalman filter
        self._motion_mat = np.eye(2 * ndim, 2 * ndim) # matrice de taille (8,8) avec des 1 sur la diagonale
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)    # matrice de taille (4,8) avec des 1 sur la diagonale


        # controle de l'incertitude de mouvement d'observation
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):

        # retourne : le vecteur mean ( dimensions 8) et la matrice de covariance (dimensions 8x8) de la nouvelle piste. Les vélocités non observées sont initialisées à 0 moyenne.
        mean_pos = measurement #coordonées de la box centre (x,y), largeur/hauteur et hauteur
        mean_vel = np.zeros_like(mean_pos) #création d'un vecteur de zero de la taille du vecteur de coordonée
        mean = np.r_[mean_pos, mean_vel] #ajoute les 2 vecteurs à la suite

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std)) #création d'une matrice avec les éléments de std sur la diagonale
        return mean, covariance

    def predict(self, mean, covariance):

        # cette fonction exécute l'étape de prédiction du filtre de Kalman.
        # mean : vecteur de la box de dimension 8 à la position précédente
        # covariance : matrice de dimension 8X8 de la box à la position précédente
        # return : le vecteur mean et la matrice de covariance à la position prédite
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) # matrice avec comme diagonale les éléments de std_pos et std_vel au carré

        mean = np.dot(self._motion_mat, mean) # calcul du nouveau vecteur mean
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov # calcul de la nouvelle matrice de covariance

        return mean, covariance

    def project(self, mean, covariance):
        # mean vecteur de dimention 8 avec les coodonées de la box
        # covariance : matrice de dimention (8x8)
        # cette fonction permet de retourner la projection du vecteur mean et de la matrice de covariance sur l'estimation

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std)) #création d'une matrice avec les éléments de std sur la diagonale

        mean = np.dot(self._update_mat, mean)   # multiplie _update_mat et mean
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) # calcul de la matice de covariance
        # on multiplie les 3 matrices ensemble
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):

        # cette focntion permet d'exécuter l'étape de correction du filtre de Kalman
        #mean : vecteur de dimension 8 de l'état
        #covariance : matrice de covariance de l'état de dimension 8x8
        #measurement : vecteur de dimension 4 avec les élémenets de la box (x, y, a, h)
        #(x, y) est le centre de la box,
        # a = largeur/hauteur
        # h = hauteur
        # return : l'état corrigé
        projected_mean, projected_cov = self.project(mean, covariance) # calcul de la projection du vecteur mean et de la matrice de covariance

        chol_factor, lower = scipy.linalg.cho_factor( projected_cov, lower=True, check_finite=False)
        # Renvoi une matrice contenant la décomposition de Cholesky,  A = U * U d'une matrice hermitienne définie positive a .
        # La valeur de retour peut être directement utilisée comme premier paramètre de cho_solve.
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T,check_finite=False).T
        #résout l'équation linéaires A x = b, étant donné la factorisation de Cholesky de A.
        # A : (chol_factor, lower)
        # b : np.dot(covariance, self._update_mat.T).T
        # x : kalman_gain
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T) # calcul du nouveau vecteur mean
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T)) #calcul de la nouvelle matrice de covariance
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,only_position=False):

        # mean vecteur de dimention 8
        # covariance : matrice de dimension (8x8)
        # measurements : matrice de dimension Nx4 (N measurements) avec les coodonnées x,y du centre , rapport largeur/hauteur et hauteur
        # only_position : si true  le calcul de la distance se fait par rapport à la position centrale de la boîte uniquement.
        # return : un vecteur de longueur N, où le i-ème élément contient le
        # distance de Mahalanobis au carré entre (moyenne, covariance) et
        # `mesures [i]`.
        mean, covariance = self.project(mean, covariance) #utilisation de la fonction project
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)# Renvoi la décomposition de Cholesky
        d = measurements - mean # calcul la différence entre measurements et mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False,overwrite_b=True)
        # résout l'équation cholesky_factor*x=d.T pour x , en supposant que cholesky_factor est une matrice triangulaire
        squared_maha = np.sum(z * z, axis=0) # fait la somme du vecteur z * z
        return squared_maha