import colorsys


import numpy as np
from keras import backend as K

from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, body_yolo, tiny_body_yolo
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

class YOLO(object):
    _base = {                                           #création d'un dico pour associer les différents éléments
        "model_path": 'model_data/yolo.h5',                 #chemin pour le modèle
        "anchors_path": 'model_data/yolo_anchors.txt', #chemin pour les anchors
        "classes_path": 'model_data/coco_classes.txt',      #chemin pour les classes
        "score": 0.3,                                       #donne une valeur de base au score
        "iou": 0.45,
        "model_image_size": (416, 416),                     #donne la dimention de l'image
        "gpu_num": 1,                                       #donne un numéro au gpu

    }
    @classmethod
    def get_base(cls, n):
        if n in cls._base:
            return cls._base[n]
        else:
            return "Nom d'attribut non reconnu'" + n + "'"

    def __init__(self,**kwargs): #le constructeur
        #La méthode update () met à jour le dictionnaire avec les éléments d'un autre dictionnaire
        self.__dict__.update(self._base) #mise à jour avec les éléments de défauts
        self.__dict__.update(kwargs)
        self.name_class = self._getclass() #obtention des noms des classes grâce à la fonction _getclass()
        self.anchors = self._getanchors()  #obtention des données des anchors grâce à la fonction _getanchors()

        self.sess = K.get_session()        #création d'un session avec Keras
        self.boxs, self.scores, self.classes = self.creation() #permet d'avoir les boxs, les scores et classes grâce à la fonction création()

    # permet de lire le fichier avec les anchors et d'enregistrer les valeurs

    def _getanchors(self):

        anchors_path = os.path.expanduser(self.anchors_path) # enregistrement de la valeur du nom d'anchors
        with open(anchors_path) as f:                        # ouverture du fichier
            anchors = f.readline()                           # lecture de la ligne
        anchors_vec = [float(j) for j in anchors.split(',')] # sépare les valeurs à chaque virgule et les transforme en float
        return np.array(anchors_vec).reshape(-1,2)           # redimensionne pour associer chaque duo de valeurs


    def creation(self):
        model_path = os.path.expanduser(self.model_path) #lecture du nom du fichier
        assert model_path.endswith('.h5'), 'Le modèle ou les poids Keras doivent être un fichier .h5.' #vérifier que fichier fini bien par .h5

        anchors_taille = len(self.anchors) # le nompbre d'anchors
        classes_taile = len(self.name_class) # le nombre de classes
        version_tiny = anchors_taille == 6    #lorsque c'est la version tiny ==> le nombre d'anchors est égale à 6

        try:
            self.model_yolo = load_model(model_path, compile=False) #chargement du modèle grâce à la fonction de Keras
        except:
            self.model_yolo = tiny_body_yolo(Input(shape=(None,None,3)), anchors_taille//2, classes_taile) \
                if version_tiny else  body_yolo(Input(shape=(None,None,3)),anchors_taille//3,classes_taile)
            self.model_yolo.load_weights(self.model_path)  # s'assurer que le modèle, les ancres et les classes correspondent

        else:
            assert self.model_yolo.layers[-1].output_shape[-1] == \
                anchors_taille/len(self.model_yolo.output) * (classes_taile +5), \
                'Inadéquation entre le modèle et les tailles d anchors et de classe données' #vérifie que la taille des anchores et des classes du modèle correspond bien aux éléments donné

        print('{} model, anchors, et classes sont chargées.'.format(model_path)) #afficher que le modèle, les ancors et les classes sont bien importés

        # Génère des couleurs pour dessiner des cadres de délimitation.
        hsv_tuples = [(x / len(self.name_class), 1., 1.)for x in range(len(self.name_class))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list( map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),self.colors))

        np.random.seed(10101)  # Correction des couleurs cohérentes entre les courses.
        np.random.shuffle(self.colors)  # Mélanger les couleurs pour décorréler les classes adjacentes.
        np.random.seed(None)  # Réinitialiser la valeur par défaut.


        self.input_image_shape = K.placeholder(shape=(2,)) # création d'un tenseur  pour filtrer les boxs

        if self.gpu_num >= 2: #si il y a un gpu
            self.yolo_model = multi_gpu_model(self.model_yolo, gpus=self.gpu_num) #utilisation du gpu pour le model yolo
        boxes, scores, classes = yolo_eval(self.model_yolo.output, self.anchors,len(self.name_class), self.input_image_shape,score_threshold=self.score, iou_threshold=self.iou) #utilisation de la fonction eval

        return boxes, scores, classes

    def image_detection(self, img):

        # cette fonction permet de détecter les objets sur l'image
        if self.model_image_size != (None,None):  #si la taille d'image n'est pas nulle
            assert self.model_image_size[0]%32 == 0, 'Multiple de 32 nécessaire'   #vérifie que la taille de l'image est bien un multiple de 32
            assert self.model_image_size[1] % 32 == 0, 'Multiple de 32 nécessaire'
            box = letterbox_image(img, tuple(reversed(self.model_image_size))) #utilisation de la fonction letterbox_image de yolo3.utile pour avoir la box


        else: #si la taille de l'image est nulle
            new_img =(img.width - (img.width % 32),img.height - (img.heaight % 32)) #création de la nouvelle image
            box = letterbox_image(img, new_img) #utilisation de la fonction letterbox_image de yolo3.utile pour avoir la box
        img_data = np.array(box, dtype='float32')  #création d'un vecteur avec les données de box

        #print(img_data.shape)
        img_data /= 255.
        img_data = np.expand_dims(img_data, 0)

        #permet d'obtenir les différentes boxs , les scores et les classes preditent  des objets
        out_boxes, out_scores, out_classes = self.sess.run( [self.boxs, self.scores, self.classes],
            feed_dict={self.model_yolo.input: img_data,self.input_image_shape: [img.size[1], img.size[0]],K.learning_phase(): 0})



         
        # création d'un vecteur pour retourner les valeurs
        return_boxs = []
        return_class_name = []
        return_score = []


        for i, p in reversed(list(enumerate(out_classes))):
            #enumerate permet d'associer un numéro à chaque valeur de out_classes (i) et p est la valeur
            classes_predict = self.name_class[p] #enregistre le nom de la classe par rapport à la valeur p

            if classes_predict != 'person': # si la classe n'est pas de type person
                continue


            box = out_boxes[i] #enregistre les valeurs de la box
            score = out_scores[i] #enregistre le score de prediction

            x = int(box[1])   #x,y point en haut à gauche de la box
            y = int(box[0])
            w = int(box[3] - box[1])
            h = int(box[2] - box[0])
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0
            return_boxs.append([x, y, w, h]) # ajoute les valeurs de la box

            return_class_name.append([classes_predict]) #ajoute le nom de la classe
            return_score.append([score]) #ajoute le score



        return return_boxs, return_class_name,return_score


    # permet de lire le fichier avec les classes et d'enregistrer les valeurs
    def _getclass(self):
        classes_path = os.path.expanduser(self.classes_path) #enregistre le nom des fichiers des classes
        with open(classes_path) as f: #ouverture du fichier et lecture grâce à la commande f
            name = f.readlines()     #lecture des lignes du fichier composé des classes
        name_vec = [i.strip() for i in name] #création d'un vecteur qui enregistre les noms des classes
        return name_vec

    def close(self):
        self.sess.close()  # permet de fermer la session
