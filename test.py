'''
PyPower Projects
Emotion Detection Using AI
'''

#USAGE : python test.py

from keras.models import load_model       # Importing all the filess for data analysis and data utilisation 
from time import sleep

from keras.preprocessing.image import img_to_array # Importing deep learning libraries 
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml') # initialize face classifier
classifier =load_model('./Emotion_Detection.h5') # predicting the emotion 

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0) # video capture if 0 is passed in the argument, it will access the web cam  



while True:  # will run forever untill we stop the execution 

    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # converting into grayscale 
    faces = face_classifier.detectMultiScale(gray,1.3,5) # passing grayscale image into the face classifier the detect mutiscale will detect the face 

    for (x,y,w,h) in faces: # these 4 variables are the coordinates of the face 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) # this 255,0,0 is the colour of the box and can be changed accordingly rest all we rae defining region of interest i.e. where the face actually is at 

        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA) # resize to 48,48 as we are using mobile net achitecture we have to use these values only to input the image   


        if np.sum([roi_gray])!=0:  # converting image into the array 
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]  # in this array there will be the probability of choosing the emotion and we have to choose the probablity that which the most suitable emotion for this face 

            print("\nprediction = ",preds)
            label=class_labels[preds.argmax()] # argmax will return the index of max probability 
            print("\nprediction max = ",preds.argmax())
            print("\nlabel = ",label)
            label_position = (x,y)

            #these four lines are responsible for the whole prediction task  

            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)  
            # putting the text on screen that this is the particular emotion detected 
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    cv2.imshow('Emotion Detector',frame)  # this i am show function is printing everything on the screen like the rectangle, detecting face, predict the emotion, put the label on the image 
    if cv2.waitKey(1) & 0xFF == ord('q'):  # to terminate press the small q key 
        break

cap.release()  # releasing the task and destroying the other windows opened in the project 
cv2.destroyAllWindows()


























