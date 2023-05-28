from PIL import Image
import cv2
import numpy as np
import pickle
from keras.models import load_model

# for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# resolution of the webcam
screen_width = 1280       # try 640 if code fails
screen_height = 720

# size of the image to predict
image_width = 224
image_height = 224

# load the trained model
model = load_model('face-detection\\test_face_cnn_model.h5')

# the labels for the trained model
with open("face-detection\\face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {key:value for key,value in og_labels.items()}
    print(labels)


# default webcam
stream = cv2.VideoCapture(0)

while(True):
    # capture frame-by-frame
    (grabbed, frame) = stream.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # try to detect faces in the webcam
    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=5)

    # for each faces found
    for index, (x, y, w, h) in enumerate(faces):

        # Draw a rectangle around the face
        color = (0, 255, 255) # in BGR
        stroke = 5
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)


        # resize the image
        size = (image_width, image_height)
        resized_image = cv2.resize(crop_img, size)
        image_array = np.array(resized_image, "uint8")
        img = image_array.reshape(1,image_width,image_height,3) 
        img = img.astype('float32')
        img /= 255

        # predict the image
        predicted_prob = model.predict(img)
        print(predicted_prob)

        # Display the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        if max(predicted_prob[0]) >= 0.7:
            name = labels[predicted_prob[0].argmax()]
        else:
            name = "unknown"
        color = (255, 0, 255)
        stroke = 2
        cv2.putText(frame, f'({name})', (x+4,y+24),
            font, 0.5, color, stroke, cv2.LINE_AA)


        cv2.imwrite(f"face-detection/image_{index + 1}.png", crop_img)

    # show the frame
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):    # Press q to break out
        break                  # of the loop

# cleanup
stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
