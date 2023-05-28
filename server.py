from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from flask_cors import CORS
import base64
import json
import pyrebase
import os
from PIL import Image
from flask_cors import CORS, cross_origin
import numpy as np
import pickle
from keras.models import load_model
import cv2


app = Flask(__name__)
app.config['CORS_HEADERS']='Content-Type'
CORS(app, expose_headers='Authorization')
run_with_ngrok(app)
app.debug=True

with open('config.json') as json_file:
    config = json.load(json_file)

firebase = pyrebase.initialize_app(config)
db = firebase.database()

model = load_model("test_face_cnn_model.h5")

image_dims = [224, 224]

with open("face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {key:value for key,value in og_labels.items()}
    print(labels)

def get_and_process(image):
    size = (image_dims[0], image_dims[1])
    resized_image = image.resize(size)
    image_array = np.array(resized_image, "uint8")
    img = image_array.reshape(1,image_dims[0],image_dims[1],3) 
    img = img.astype('float32')
    img /= 255
    image.close()
    resized_image.close()
    return img

@app.route("/get_img", methods = ['GET', 'POST'])
def recieve():
  data = request.get_json()

  images = data['images']
  time = data['time']
  paths = data['paths']

  preds = {}

  for i in range(len(paths)):
    string_bytes = bytes(str(images[i]), 'utf-8')

    with open(paths[i], "wb") as fh:
      fh.write(base64.decodebytes(string_bytes))
    
    img = get_and_process(Image.open(paths[i]))
    predicted_prob = model.predict(img)
    print(predicted_prob)

    if max(predicted_prob[0]) >= 0.7:
      name = labels[predicted_prob[0].argmax()]
    else:
      name = "unknown"

    db.child(f'data/{time}').update(
                {i: {
                    'img': str(images[i]),
                    'whoIsIt': name
                }}
            )

    preds[f'{i}'] = name

  return json.dumps(preds)


if __name__ == "__main__":
  app.run()