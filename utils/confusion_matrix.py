import os

import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import model_functions as mf


def show_confusion_matrix():
    model = mf.load_model()
    truth = []
    predictions = []
    test_directory = "../test_data/"
    labels = sorted([name for name in os.listdir(test_directory) if os.path.isdir(os.path.join(test_directory, name))])
    for i, font in enumerate(labels):
        font_path = os.path.join(test_directory, font)
        for filename in os.listdir(font_path):
            f = os.path.join(font_path, filename)
            truth.append(i)
            this_image = mf.load_image(f)
            this_prediction = mf.predict_from_image(this_image, model)
            this_prediction = np.argmax(this_prediction)
            predictions.append(this_prediction)
            if this_prediction != i:
                print(f, labels[this_prediction])

    conf_matrx = confusion_matrix(y_true=truth, y_pred=predictions)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.matshow(conf_matrx, cmap=matplotlib.cm.get_cmap('tab20c'), alpha=0)

    for col in range(conf_matrx.shape[0]):
        for row in range(conf_matrx.shape[1]):
            ax.text(x=row, y=col, s=conf_matrx[row, col], va='center', ha='center', size='xx-large')
    plt.xlabel("Predictions", fontsize=18)
    labels.insert(0, "")
    ax.xaxis.set_ticklabels(labels)
    ax.xaxis.set_tick_params(which='both', top=False, labeltop=False, labelbottom=True)
    ax.yaxis.set_ticklabels(labels)
    plt.ylabel("Actual", fontsize=18)
    plt.title("Confusion Matrix", fontsize=18)
    plt.savefig('confusion_matrix.png')
    plt.show()


show_confusion_matrix()

