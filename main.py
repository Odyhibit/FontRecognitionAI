import os
import numpy as np
import tensorflow
from tensorflow import keras as keras


def predict_from_image(test_image: np.array, this_model: keras.models) -> np.array('float'):
    return this_model.predict(test_image, verbose=0)


def load_image(image_file: str):
    img_array = keras.utils.load_img(image_file, target_size=(224, 224))
    img_array = keras.utils.img_to_array(img_array, dtype='float32')
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_predictions(predictions: np.array('float')):
    # print(np.argmax(predictions), get_labels())
    font_name = get_labels()[np.argmax(predictions)]
    accuracy = predictions[np.argmax(predictions)]
    return font_name + " " + str(round(accuracy * 100, 2)) + "%"


def get_labels():
    labels = [name for name in os.listdir("./test_data/") if os.path.isdir(os.path.join("./test_data/", name))]
    return sorted(labels)


def main(files_to_match: [], model):
    for img in files_to_match:
        img_batch = load_image(img)
        prediction = predict_from_image(img_batch, model)
        print("Raw prediction numbers", prediction[0])
        print(get_predictions(prediction[0]))


def load_model():
    tensorflow.get_logger().setLevel('NONE')
    model_file = 'EfficientNetV2B1_model'
    model = keras.models.load_model(model_file)
    return model

if __name__ == '__main__':
    user_file_lst = ['user_files/test_img.jpg', 'user_files/Ancient_times.jpg', 'user_files/Ancient_cour.jpg']
    main(user_file_lst, load_model())
