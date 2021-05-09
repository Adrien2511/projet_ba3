
import numpy as np
import cv2
import tensorflow as tf

# ce fichier va permettre d'extraire les caractéristiques des images comprise dans les boxs afin de pouvoir appliquer le tracking
def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out) #taille de l'élément out
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):


    #cette fonction permet de retrouner l'image de la box
    #image : l'image en entiére
    #bbox : la box dans le  format (x, y, width, height)
    #patch_shape : ce paramètre permet de donner des dimensions souhaitée , ce qui va donc modifier les dimensions de la box
    bbox = np.array(bbox) # création d'un vecteur avec les coordonées de la box, les coordonées sont en format (x,y, width, height)
    if patch_shape is not None: # si il y a un élément patch_shape

        # rapport hauteur/ largeur à la forme du patch
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3] #calcul de la nouvelle largeur
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width #enregistrement de la nouvelle largueur


    bbox[2:] += bbox[:2] #transforme les coodonées en coin  haut gauche (x,y) et bas droite (x,y)
    bbox = bbox.astype(np.int) # transormation en entier

    # clip aux limites de l'image
    bbox[:2] = np.maximum(0, bbox[:2]) #prend les valeurs de x,y de bbox si elles sont positives
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]): #si la box est vide
        return None
    sx, sy, ex, ey = bbox # enregistre les valeurs
    image = image[sy:ey, sx:ex] # enregistrement de la partie de l'image où se situe la box
    image = cv2.resize(image, tuple(patch_shape[::-1])) # resize de l'image au format voulu afin d'avoir que l'image de la box
    return image


class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",output_name="features"): #constructeur de la classe
        self.session = tf.Session() #création d'une session de tensorflow
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            #transformation du message en message binaire
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())

        tf.import_graph_def(graph_def, name="net") #importation du nouveau message
        self.input_var = tf.get_default_graph().get_tensor_by_name("net/%s:0" % input_name) #obtention du nom du tensor d'input
        self.output_var = tf.get_default_graph().get_tensor_by_name("net/%s:0" % output_name) #obtention du nom du tensor d'output

        assert len(self.output_var.get_shape()) == 2  #vérification de la taille de output_var
        assert len(self.input_var.get_shape()) == 4   #vérification de la taille de input_var
        self.feature_dim = self.output_var.get_shape().as_list()[-1] #enregistre une liste d'intenger  du dernier élément
        self.image_shape = self.input_var.get_shape().as_list()[1:]  #enregistre une liste d'intenger de tout les éléments sauf le premier

    def __call__(self, data_x, batch_size=32): #fonction appeller durant l'utilisation de l'objet de la classe ImageEncoder
        out = np.zeros((len(data_x), self.feature_dim), np.float32) # création d'une matrice de zeros de taille (nombre d'image de box , feature_dime)

        _run_in_batches(lambda x: self.session.run(self.output_var, feed_dict=x),{self.input_var: data_x}, out, batch_size)
        return out


def create_box_encoder(model_filename, input_name="images",output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name) #création d'un objet de la classe Image encoder
    image_shape = image_encoder.image_shape #enregistrement de la taille de l'image

    def encoder(image, boxes):
        image_patches = []
        for box in boxes: # pour chaque élément du vecteurs boxes qui sont en format (x, y, width, height)
            patch = extract_image_patch(image, box, image_shape[:2]) #permet d'obtenir l'image de la box
            if patch is None: # si pas d'élément
                print("WARNING: Échec de l'extraction du patch d'image: %s." % str(box)) # affichage d'une erreur et ajout d'un élément random
                patch = np.random.uniform( 0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch) #ajout de l'image à la liste des toutes les images de boxs
        image_patches = np.asarray(image_patches) #création d'un vecteur
        return image_encoder(image_patches, batch_size) #utilisation de la fct __call__ de la classe ImageEncoder

    return encoder #appele de la fonction encoder
