#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np


# parameters for loading data and images
detection_model_path = './haarcascade_frontalface_default.xml'
emotion_model_path = './_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["Angry" , "Happy", "Sad", "Surprised","Neutral"]
COLOR    = [(0,0,255), (0,255,0), (0,165,255), (255,0,0), (127,127,127)]
preds1    =[0,0,0,0,0]

# starting video streaming
cv2.namedWindow('LIVE_CAMERA')
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((200, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds    = emotion_classifier.predict(roi)[0]
        preds[4] = np.maximum(preds[1],preds[4])
        preds[5] = np.maximum(preds[2],preds[5]) 
        preds    = np.delete(preds,[1,2])
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        box_colour = COLOR[preds.argmax()]
        for (i, (emotion, prob ,tempColor)) in enumerate(zip(EMOTIONS, preds,COLOR)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                
                w = int(prob * 300)
                
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), tempColor, -1)
                
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)
                
                cv2.rectangle(frameClone, (fX-1, fY-25), (fX + fW+1, fY ),
                              box_colour, -1)
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX  , 0.45, (255, 255, 255), 1)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              box_colour, 2)
        cv2.imshow('LIVE_CAMERA', frameClone)
        cv2.imshow("Probabilities", canvas)
        
    else: 
            for (i, (emotion, prob ,tempColor)) in enumerate(zip(EMOTIONS, preds1,COLOR)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, 0 * 100)
                
                w = int(prob * 300)
                
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (0, (i * 35) + 35), tempColor, -1)
                
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)
                
            cv2.imshow('LIVE_CAMERA', frame)
            cv2.imshow("Probabilities", canvas)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




