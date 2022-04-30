import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    
    img = cv2.resize(frame, (224,224))
    test_img = np.array(img, dtype= np.float32)
    test_img = np.expand_dims(test_img, axis= 0)
    normalize_img = test_img/255.0
    prediction = model.predict(normalize_img)
    print(prediction)

    cv2.imshow('Result', frame)
    if cv2. waitKey(23) == 32:
        break

video.release()
cv2.destroyAllWindows()

