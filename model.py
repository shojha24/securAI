import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras_vggface.vggface import VGGFace
from keras import Model
import pickle
from PIL import Image

if tf.test.is_gpu_available():
    print("GPU is available and enabled for TensorFlow.")
else:
    print("GPU is NOT available or not enabled for TensorFlow.")


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
'images',
target_size=(224,224),
color_mode='rgb',
batch_size=32,
class_mode='categorical',
shuffle=True)

train_generator.class_indices.values()
NO_CLASSES = len(train_generator.class_indices.values())


base_model = VGGFace(include_top=False,
model='vgg16',
input_shape=(224, 224, 3))
base_model.summary()
print(len(base_model.layers))
# 19 layers after excluding the last few layers


x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)

# final layer with softmax activation
preds = Dense(NO_CLASSES, activation='softmax')(x)

# create a new model with the base model's original input and the 
# new model's output
model = Model(inputs = base_model.input, outputs = preds)
model.summary()

# don't train the first 19 layers - 0..18
for layer in model.layers[:19]:
    layer.trainable = False

# train the rest of the layers - 19 onwards
for layer in model.layers[19:]:
    layer.trainable = True

model.compile(optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

'''try:
    for path in train_generator.filepaths:
        try:
            image = Image.open(path)  # Use Image module to load image
            # Perform further processing on the image
        except Exception as e:
            print(f"Error opening image at path: {path}")
            print(f"Error details: {e}")
except AttributeError:
    print("train_generator doesn't have filepaths attribute.")'''

try: 
    model.fit(train_generator,
  batch_size = 1,
  verbose = 1,
  epochs = 5)
except Exception as e:
    print(f"Error occurred during training: {e}")


# creates a HDF5 file
model.save('test_face_cnn_model.h5')

class_dictionary = train_generator.class_indices
class_dictionary = {
    value:key for key, value in class_dictionary.items()
}
print(class_dictionary)

face_label_filename = 'face-labels.pickle'
with open(face_label_filename, 'wb') as f: 
    pickle.dump(class_dictionary, f)

