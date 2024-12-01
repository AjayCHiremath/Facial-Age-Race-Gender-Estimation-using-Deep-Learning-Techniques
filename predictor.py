import cv2
from tensorflow.keras.models import load_model
import numpy as np
gender_str = {0: "Male", 1: "Female"}

def predict(im):
    im = cv2.resize(im, (198,198),interpolation = cv2.INTER_NEAREST)
    im = np.expand_dims(im/255.0,axis=0)

    print(im.shape)
    # return
    model = load_model("best_model.h5")
    age,race,gender = model.predict(im)
    # print(a,g,r)
    print(age*116, race.argmax(axis=-1), gender_str.get(gender.argmax(axis=-1)[0]))
    return age*116, gender_str.get(gender.argmax(axis=-1)[0])