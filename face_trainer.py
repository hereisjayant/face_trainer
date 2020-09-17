import cv2
import numpy as np
from PIL import Image
import os

#path to the images collected
path = 'data_sets'
                      #LOCAL BINARY PATTERNS HISTOGRAMS
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

#function to get images and data labels:
def getImagesAndLabels(path):
    #creates an array of img paths by joining 'path' and the img path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids= []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        #crates an array of the image
        img_numpy = np.array(PIL_img,'uint8')
        #this gets the user ID by splitting filename
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        #detects the face in the image
        for (x,y,w,h) in faces:
            #add face to the faces array
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            #add the ID corresponding to the face array
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.save('trainer.yml')
#Prints the no of faces trained
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


    
        
