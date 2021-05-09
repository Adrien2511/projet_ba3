from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np

from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from collections import deque
from keras import backend

backend.clear_session()


chemin_model = "model_data/yolo.h5"  # on encode les chemins d'accès au modèle et à la vidéo "test"




classes = "model_dat/coco_classes.txt"  # on se base sur le modèle "coco" pour nos différentes classes

# enregistrement des cadres ("anchors") --> contour des boxs


anchors_yolo = "model_data/yolo_anchors.txt" # chemin pour les anchors


# enregistrement du modèle
yolo = YOLO(classes_path1=classes, anchors_path1=anchors_yolo, model_path1=chemin_model)


warnings.filterwarnings('ignore')


np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8") # initialiser une liste de couleurs

def intersect(A,B,C,D): # fonction qui permet de savoir si il y a une intersection
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C): # calcul pour l'intersection
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def main(yolo):


    #Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.4 #donne la superposition maximale des boxs

    counter = 0 #création du compteur
    memory={}

    import socket
    # ce bloc permet de faire la connexion udp en multicast
    group = '239.255.255.65'
    port = 6539

    ttl = 2
    sock = socket.socket(socket.AF_INET,
                         socket.SOCK_DGRAM,
                         socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP,
                    socket.IP_MULTICAST_TTL,
                    ttl)




    model_filename = 'model_data/market1501.pb' #model qui permet d'enregistrer les caractéristiques comprise dans les boxs
    encoder = gdet.create_box_encoder(model_filename,batch_size=1) #utilisation de la fonction create_box_encoder du fichier generate_detection afin de pouvoir plus tard encoder les boxs et les images

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)  # création d'un objet de la classe NearestNeighborDistanceMetric du fichier nn_matching
    # qui va permettre de calculer les distances
    tracker = Tracker(metric) #création d'un élément de la class Tracker du fichier tracker



    video_file="videoKang.avi"  #chemin pour la vidéo
    #video_capture = cv2.VideoCapture('rtsp://admin:admin@192.168.0.100:554/ch0_0.264')  #ouverture de la vidéo
    video_capture = cv2.VideoCapture(video_file) #ouverture de la vidéo
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # initialisation de la largeur de la video : 640px
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


    # Define the codec and create VideoWriter object
    w = int(video_capture.get(3)) #prise de la largeur de la vidéo
    h = int(video_capture.get(4)) #prise de la hauteur de la vidéo
    #w = 480;
    #h = 360;
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('testyolo24.AVI', fourcc, 10, (w,h)) #création d'un élément pour pouvoir enregistrer
    #line = [(int(w/6), int(3*h/8)), (int(4*w/6), int(5*h/6))] # création des différentes lignes pour le comptage
    #line2 = [(int(w/5), int(1*h/13)),(int(w/6),int(3*h/8))]
    #line3 =[(int(4.5*w/6), int(3*h/7)),(int(4*w/6), int(5*h/6))]

    #line = [(int(w/6), int(4*h/6)), (int(4*w/6), int(4*h/6))] # création des différentes lignes pour le comptage
    #line2 = [(int(w/6), int(1*h/13)),(int(w/6),int(3*h/8))]
    #line3 =[(int(4.5*w/6), int(3*h/7)),(int(4*w/6), int(4*h/6))]

    line = [(int(w/10), int(3 * h / 6)),(int(5 * w / 6), int(3 * h / 6))]  # création des différentes lignes pour le comptage
    line2 = [(int(w/10), int(8*h/9)),(int(w/10),int(3*h/6))]
    line3 =[(int(5*w/6), int(8*h/9)),(int(5*w/6), int(3*h/6))]



    frame_index = -1

    fps = 0.0  # création d'une variable pour calculer les fps
    totalframe = 0 #création d'une variable pour pouvoir n'utiliser que certaines frames


    while True:

            ret, frame = video_capture.read()  # lecture de la vidéo


            if ret != True: #vérifier si la vidéo est bien ouverte
             break

        #if totalframe%2 == 0: #utilisation d'un modulo pour n'utiliser que certaines frames

            t1 = time.time()


            #frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA) #resize de l'image

            image = Image.fromarray(frame[...,::-1]) #transforme l'image en noir et blanc
            boxs,class_names, return_score = yolo.image_detection(image) #renvoi les coordonées de la box , la class et le score

            features = encoder(frame,boxs) # générer des caractéristiques pour la ré-identification des personnes
            #,permettant de comparer l'apparence visuelle des boîtes de délimitation des personnes en utilisant la similarité du cosinus.

            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)] # création des détections

            boxes = np.array([d.tlwh for d in detections]) #enregistre les boxs
            scores = np.array([d.confidence for d in detections]) #enregistre les scores

            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores) #retourne les indices des boxs garder
            detections = [detections[i] for i in indices] #garde que les éléments qui ont leur indice dans le vecteur indice


            tracker.predict() #appelle de la fonction predict de la classe tracker
            tracker.update(detections) #mise à jour des détections

            i = int(0)
            indexIDs = [] # création d'un vecteur pour enrgistrer les indices

            previous = memory.copy() # copy du dictionnaire précédent
            memory={} # remise à 0 du dictionnaire

            for track in tracker.tracks: #appliquer pour chaque élément tracker
                if not track.is_confirmed() or track.time_since_update > 1: # si pas de détection
                    continue

                indexIDs.append(int(track.track_id)) #enregistrement de l'indice




                bbox = track.to_tlbr()  #les coordonées de la box en : (min x, min y, max x, max y)
                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]] # enregistre une couleur

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3) #création du cadre
                cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2) #écriture  de l'id sur la box
                if len(class_names) > 0:

                   cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2) #écriture du nom de la classe

                i += 1
                #bbox_center_point(x,y)
                #center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2)) # calcul le centre de la box
                center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3]))/1.3 ))  # calcul le centre au niveau des pieds de la personne

                memory[indexIDs[-1]] = center # enregistre dans le dictionnaire l'id et le centre de la box
                #track_id[center]

                thickness = 5 # la taille de l'élément au centre

                cv2.circle(frame,  (center), 1, color, thickness) #crée l'élément au centre du cadre









            for i in range(len(indexIDs)): # boucle sur chacun des indices des éléments trackés afin de savoir si ils ont passé la ligne
                if indexIDs[i] in previous: # vérifie si l'indice était déjà là à la frame précédente car sinon ce n'est pas possible que la personne soit passé au dessus de la ligne
                    previous_box = previous[indexIDs[i]] #prend les coordonées du centre de l'objet tracké à la frame précédente
                    now_box = memory[indexIDs[i]]        #prend les coordonées du centre de l'objet tracké à la frame actuelle

                    p0 = (int(previous_box[0]), int(previous_box[1])) #enregistre les valeurs  du centre à la position précédente
                    p1 = (int(now_box[0]), int(now_box[1])) #enregistre les valeurs du centre à la position actuelle


                    if intersect(p0, p1, line[0], line[1]): # appel de la fonction intersect
                        # elle va renvoyer true si il y une intersection entre p0 p1 et line [0] line[1]
                        if p1[1]<p0[1]: #compare la coordonée y de p1 et p0 pour savoir si il faut augmenter ou diminuer le compteur

                           counter += 1
                        else:
                            counter -=1
                        print(counter)


                        sock.sendto(counter.to_bytes(2, 'big'), (group, port)) #envoi du compteur


                    if intersect(p0, p1, line2[0], line2[1]): # regarde si il y a une intersection entre la position actuelle et précédente de la boxe et la ligne 2 comme précédement
                        if p1[0]<p0[0]:    # permet de connaitre le sens du passage pour augementer ou diminuer le compteur

                           counter += 1
                        else:
                            counter -=1
                        print(counter)

                        sock.sendto(counter.to_bytes(2, 'big'), (group, port)) #envoi du compteur

                    if intersect(p0, p1, line3[0], line3[1]): # regarde si il y a une intersection entre la position actuelle et précédente de la boxe et la ligne 3 comme précédement
                        if p0[0]>p1[0]: # permet de connaitre le sens du passage pour augmenter ou diminuer le compteur

                           counter += 1
                        else:
                            counter -=1
                        print(counter)

                        sock.sendto(counter.to_bytes(2,'big'), (group, port)) #envoi du compteur



            #cv2.putText(frame, "Total Object Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
            #cv2.putText(frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
            #cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)

            # création des lignes pour avoir un indicateur visuelle
            cv2.line(frame, line[0], line[1], (0, 255, 255), 3)
            cv2.line(frame, line2[0], line2[1], (0, 255, 255), 3)
            cv2.line(frame, line3[0], line3[1], (0, 255, 255), 3)


            cv2.putText(frame,str(counter), (int(30), int(100)), 0, 5e-3 * 400, (0, 255, 0), 5)#écriture du compteur sur la video
            cv2.namedWindow("YOLO", 0); #donne le nom de la fenêtre
            cv2.resizeWindow('YOLO', 640, 360); #donne la taille de la fenêtre
            cv2.imshow('YOLO',frame) #affiche le résultat




            out.write(frame) #enregistre la frame
            frame_index = frame_index + 1 #augmente le nombre de frame

            fps  = ( fps + (1./(time.time()-t1)) ) / 2 #calcul des fps



            totalframe += 1 #augmente le compteur de frame
            if cv2.waitKey(1) & 0xFF == ord('q'):  #permet de fermer la fenêtre
                break
    print(" ")
    print("[Finish]")



    video_capture.release() # fermeture de la vidéo


    
    out.release() # fermeture de l'enregistrement

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())