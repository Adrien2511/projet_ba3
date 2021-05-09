
import subprocess
# ce fichier permet de lancer le fichier convert.py et video.py dans un seul fichier
class facade:       #création de la classe façade
    def __init__(self,model):  #constructeur
        while True:

            if model=="oui": #permet de choisir d'enrégister le modèle ou pas
                print("Model normal(1) ou tiny(2) ?")
                tiny=input()
                if tiny=="2":
                    subprocess.call(" python convert.py yolov3-tiny.cfg yolov3-tiny.weights model_data/yolo.h5 ", shell=True) #lance la création du modèle en tiny
                    subprocess.call("python video.py", shell=True) # lance la détection et le comptage vidéo
                    break
                elif tiny=="1":
                    subprocess.call(" python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5 ", shell=True) #lance la création du modèle
                    subprocess.call("python video.py", shell=True) # lance la détection et le comptage vidéo
                    break

                else:
                    print("réponse impossible")

            elif model=="non":

                subprocess.call("python video.py", shell=True) # lance la détection et le comptage vidéo
                break
            else:
                print("réponse impossible")

if __name__ == '__main__':
    print("Enregister le model (oui ou non)  ?")
    model = input()
    facade(model)