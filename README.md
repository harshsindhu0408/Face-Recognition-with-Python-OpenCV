# Face-Recognition-with-Python-OpenCV
In this deep learning project, we will learn how to recognize the human faces in live video with Python.

About dlib’s Face Recognition:
Python provides face_recognition API which is built through dlib’s face recognition algorithms. This face_recognition API allows us to implement face detection, real-time face tracking and face recognition applications.

Project Prerequisites:
You need to install the dlib library and face_recognition API from PyPI:

Steps to implement Face Recognition with Python:
We will build this python project in two parts. We will build two different python files for these two parts:

embedding.py: In this step, we will take images of the person as input. We will make the face embeddings of these images.
recognition.py: Now, we will recognize that particular person from the camera frame.

1. embedding.py:

First, create a file embedding.py in your working directory. In this file, we will create face embeddings of a particular human face. We make face embeddings using face_recognition.face_encodings method. These face embeddings are a 128 dimensional vector. In this vector space, different vectors of same person images are near to each other. After making face embedding, we will store them in a pickle file.

Import necessary libraries:
import sys
import cv2 
import face_recognition
import pickle
To identify the person in a pickle file, take its name and a unique id as input:

name=input("enter name")
ref_id=input("enter id")
Create a pickle file and dictionary to store face encodings:
try:
    f=open("ref_name.pkl","rb")
    ref_dictt=pickle.load(f)
    f.close()
except:
    ref_dictt={}
ref_dictt[ref_id]=name
f=open("ref_name.pkl","wb")
pickle.dump(ref_dictt,f)
f.close()
try:
    f=open("ref_embed.pkl","rb")
    embed_dictt=pickle.load(f)
    f.close()
except:
    embed_dictt={}
    
Open webcam and 5 photos of a person as input and create its embeddings:
Here, we will store the embeddings of a particular person in the embed_dictt dictionary. We have created embed_dictt in the previous state. In this dictionary, we will use ref_id of that person as the key.

To capture images, press ‘s’ five times. If you want to stop the camera press ‘q’:

Update the pickle file with the face embedding.

2. recognition.py:

Here we will again create person’s embeddings from the camera frame. Then, we will match the new embeddings with stored embeddings from the pickle file. The new embeddings of same person will be close to its embeddings into the vector space. And hence we will be able to recognize the person.

Import the libraries:
import face_recognition
import cv2
import numpy as np
import glob
import pickle
Load the stored pickle files:
f=open("ref_name.pkl","rb")
ref_dictt=pickle.load(f)        
f.close()
f=open("ref_embed.pkl","rb")
embed_dictt=pickle.load(f)      
f.close()
Create two lists, one to store ref_id and other for respective embedding:
known_face_encodings = []  
known_face_names = []  
for ref_id , embed_list in embed_dictt.items():
    for my_embed in embed_list:
        known_face_encodings +=[my_embed]
        known_face_names += [ref_id]
Start the webcam to recognize the person:
video_capture = cv2.VideoCapture(0)
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
while True  :
  
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
    process_this_frame = not process_this_frame
    for (top_s, right, bottom, left), name in zip(face_locations, face_names):
        top_s *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top_s), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, ref_dictt[name], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
Now run the second part of the project to recognize the person:
