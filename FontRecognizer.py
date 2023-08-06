
import io
import tkinter.filedialog
from tkinter import *
from tkinter import ttk

import matplotlib
import numpy as np
from PIL import Image, ImageTk, ImageOps
from matplotlib import pyplot as plt
from tensorflow import keras as keras

from utils import model_functions as mf


def pick_cover(event: Event = Event()):
    global model
    file_name = tkinter.filedialog.askopenfilename(initialdir=".\\user_files", filetypes=[("image files", ".jpg")])
    if not file_name:
        return
    image_path_str.set(file_name)
    test_img = mf.load_image(file_name)
    preview_img = Image.open(file_name)
    preview_img = ImageTk.PhotoImage(preview_img)
    preview_lbl.configure(image=preview_img)
    preview_lbl.image = preview_img
    predictions = mf.predict_from_image(test_img, model)
    predictions_to_image(predictions)


def delay_load_model(e):
    global model
    # your code here
    model = keras.models.load_model(model_file)
    root.title("Font Recognizer")
    root.unbind('<Visibility>')  # only call `callback` the first time `root` becomes visible


def predictions_to_image(prediction):
    matplotlib.use("Agg")
    prediction_list = prediction[0].tolist()
    label_list = mf.get_labels()
    size_list = prediction_list
    sizes, labels = [], []
    for i, num in enumerate(prediction_list):
        if num >= .01:
            sizes.append(size_list[i])
            labels.append(label_list[i])
    plt.rcParams["figure.figsize"] = [3.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    # plt.rcParams.update({'font.size': 22})
    sizes = np.array(sizes)
    percent_labels = 100. * sizes / sizes.sum()
    colors = ['paleturquoise', 'lightsteelblue', 'lavender', 'lightgrey', 'lightskyblue', ]
    # fig, ax = plt.subplots()

    # ax.pie(sizes, labels=labels)
    patches, texts = plt.pie(sizes, colors=colors, startangle=225, radius=1.2)
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(labels, percent_labels)]
    patches, labels, dummy = zip(*sorted(zip(patches, labels, sizes), key=lambda x: x[2], reverse=True))
    plt.legend(patches, labels, loc='center left', bbox_to_anchor=(-0.1, 1.), fontsize=16)
    # plt.show()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')

    im = Image.open(img_buf)
    pie_chart_img = ImageOps.contain(im, (256, 256))
    pie_chart_img = ImageTk.PhotoImage(pie_chart_img)

    # im.show(title="My Image")
    output.configure(image=pie_chart_img)
    output.image = pie_chart_img
    img_buf.close()


root = Tk()
root.title("Font Recognizer - Loading AI Framework")
root.geometry("750x450")
icon_fr = PhotoImage(file="app_images/font_recognizer.png")
root.iconphoto(False, icon_fr)

# notebook settings
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill=BOTH)

# main screen variables
image_path_str = StringVar()
image_path_str.set("Font Image")

# main tab widgets
main_screen = Frame(notebook)
main_screen.columnconfigure(0, minsize=400)
main_screen.columnconfigure(1, minsize=200)

# row 0
file_lbl = ttk.Label(main_screen, textvariable=image_path_str, wraplength=400)
file_lbl.grid(column=0, row=0, padx=4, pady=6)
file_btn = Button(main_screen, text="Choose Image", command=pick_cover)
file_btn.grid(column=1, row=0, padx=4, pady=6)
# row 1
placeholder = ImageTk.PhotoImage(Image.open("app_images/placeholder.png"))
preview_lbl = Label(main_screen, bd=1, relief="solid", image=placeholder)
preview_lbl.grid(column=0, row=1, rowspan=2, sticky="S", padx=4, pady=6)
output_lbl = Label(main_screen, text="Font Certainty", height=1)
output_lbl.grid(column=1, row=1, sticky="S")
output = Label(main_screen, bd=1, relief="solid", image=placeholder)
output.grid(column=1, row=2, sticky="NSEW", padx=4, pady=6)

# dashboard tab widgets
dashboard_screen = Frame(notebook)

accuracy_img = Image.open("app_images/accuracy.png")
accuracy_img = ImageOps.contain(accuracy_img, (320, 320))
accuracy_img = ImageTk.PhotoImage(accuracy_img)
accuracy_label = Label(dashboard_screen, bd=1, relief='solid', image=accuracy_img)
accuracy_label.grid(column=0, row=0, padx=(40, 15), pady=60)
confusion_img = Image.open("app_images/confusion_matrix.png")
confusion_img = ImageOps.contain(confusion_img, (320, 320))
confusion_img = ImageTk.PhotoImage(confusion_img)
confusion_label = Label(dashboard_screen, bd=1, relief='solid', image=confusion_img)
confusion_label.grid(column=1, row=0, padx=(15, 40), pady=60)

# notebook tabs
notebook.add(main_screen, text="Main")
notebook.add(dashboard_screen, text="Dashboard")

model_file = 'EfficientNetV2B1_model'
model = None

first_time = True
if first_time:
    output.bind('<Visibility>', delay_load_model(first_time))
    first_time = False
root.mainloop()
