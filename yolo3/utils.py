from functools import reduce

from PIL import Image


def compo(*funcs):

    if funcs: #si ce n'est pas vide
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    #La fonction reduction  permet d'appliquer une fonction au paramètres f et g  de "funcs".
    else:
        raise ValueError('La composition de la séquence vide n est pas prise en charge.') #return une valeur d'erreur si il y a pas de funcs


def letterbox_image(img, size):

    w_initiale, h_initiale =img.size # on prends les dimentions de l'image de base
    w, h = size #on prends les valeurs de l'attribut size
    taille = min(w/w_initiale, h/h_initiale) #prend le min entre le rapport de largeur et de hauteur

    new_w = int(taille*w_initiale) #calcul de la nouvelle largeur
    new_h = int(taille*h_initiale) #calcul de la nouvelle hauteur

    img=img.resize((new_w,new_h), Image.BICUBIC) #redimentionne l'image
    new_img = Image.new('RGB', size, (128,128,128)) #création d'une nouvelle image
    new_img.paste(img, ((w-new_w)//2, (h-new_h)//2)) #paste permet d'intégrer l'ancienne  sur la nouvelle en donnant la position

    return new_img





