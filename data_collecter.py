import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) #setting the video width
cam.set(4, 480) #setting height

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

#asks person fthe ID
face_id = input('\n Enter the user ID then press return ===> ')
print('\n please look at the camera')
print('\n initializing...')

count = 0

while(1):
    ret, img = cam.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face = face_cascade.detectMultiScale(
        gray,
        1.2,
        5,
        minSize=(30, 30)   #minimum rectangle size ti be considered a face
        )
    
    for(x, y, w, h) in face:
        count = count+1
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255.0), 2)
        
        print('Capturing image no. '+str(count))
        #saving the image
        file_name = "data_sets/Employee."+str(face_id)+"."+str(count)+".png"
        cv2.imwrite(file_name, gray[y:y+h, x:x+w]) #gets the face in grayscale
        #showing the photo being taken
        cv2.imshow('Preview', img)
        
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
        break

#closing all windows
print("Data Collected!")
cam.release()
cv2.destroyAllWindows()
        
    
