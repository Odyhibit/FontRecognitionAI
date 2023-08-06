import os

import numpy as np
from tensorflow import keras as keras


def predict_from_image(test_image: np.array, this_model: keras.models) -> np.array('float'):
    return this_model.predict(test_image, verbose=0)


def load_image(image_file: str):
    img_array = keras.utils.load_img(image_file, target_size=(224, 224))
    img_array = keras.utils.img_to_array(img_array, dtype='float32')
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_labels():
    labels = [name for name in os.listdir("test_data\\") if os.path.isdir(os.path.join("test_data\\", name))]
    return sorted(labels)


def load_model():
    model_file = '..\\EfficientNetV2B1_model'
    model = keras.models.load_model(model_file)
    return model
