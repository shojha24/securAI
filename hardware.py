import cv2
import time
import datetime
import os
import requests
import base64
import pygame
from PIL import Image
import numpy as np
import pickle
import json

stream = cv2.VideoCapture(0)
pygame.mixer.init()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image_dim = [224, 224]

def convert_to_64(img_paths):
    encoded_imgs = []
    if not img_paths:
        pass
    else:
        for img_path in img_paths:
            with open(img_path, 'rb') as f:
                img_data = f.read()
                encoded_img = base64.b64encode(img_data).decode('utf-8')
                encoded_imgs.append(encoded_img)
    return encoded_imgs, img_paths

def camera_feed():
    list_of_images = []
    global frame
    ret, frame = stream.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # try to detect faces in the webcam
    global faces
    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=5)

    # for each faces found
    
    for index, (x, y, w, h) in enumerate(faces):

        # Draw a rectangle around the face
        color = (0, 255, 255) # in BGR
        stroke = 5
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
        
        cv2.imwrite(f"image_{index + 1}.png", crop_img)
        list_of_images.append(f"image_{index + 1}.png")

    return list_of_images

while True:
    encoded_imgs, img_paths = convert_to_64(camera_feed())
    
    url = 'http://3bc0-69-119-107-111.ngrok-free.app/get_img'
    
    now = datetime.datetime.now()
    date_time = now.strftime('%Y-%m-%d %H:%M:%S')
    
    if encoded_imgs:
        response = requests.get(url, json={"images": encoded_imgs, "time": date_time, "paths": img_paths})
        
        print(response.status_code, response.content)
        
        images = json.loads(response.content)

        for key in images:
            pred = images[key]
            (x, y, w, h) = faces[int(key)]

            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 0, 255)
            stroke = 2
            cv2.putText(frame, f'({pred})', (x+4,y+24),
            font, 0.5, color, stroke, cv2.LINE_AA)

            """if pred == "atharav":
                alert = pygame.mixer.Sound("/home/spoutshield/Documents/audio/alert_atharav.wav")
            elif pred == "mummy":
                alert = pygame.mixer.Sound("/home/spoutshield/Documents/audio/alert_mummy.wav")
            elif pred == "papa":
                alert = pygame.mixer.Sound("/home/spoutshield/Documents/audio/alert_papa.wav")
            elif pred == "raghav":
                alert = pygame.mixer.Sound("/home/spoutshield/Documents/audio/alert_raghav.wav")
            elif pred == "zac":
                alert = pygame.mixer.Sound("/home/spoutshield/Documents/audio/alert_zac.wav")
            else:
                alert = pygame.mixer.Sound("/home/spoutshield/Documents/audio/alert_intruder.wav")

            alert.play()
            while pygame.mixer.get_busy():
                pass"""
    
    cv2.imshow('frame', frame)
     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1)

stream.release()
cv2.destroyAllWindows()