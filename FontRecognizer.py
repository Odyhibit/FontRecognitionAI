import tkinter.filedialog
import io
from tkinter import *
from tkinter import ttk
import matplotlib

from matplotlib import pyplot as plt

from utils import model_functions as mf
from tensorflow import keras as keras
from PIL import Image, ImageTk, ImageOps


def pick_cover(event: Event = Event()):
    global model
    file_name = tkinter.filedialog.askopenfilename(initialdir="./test_images", filetypes=[("image files", ".jpg")])
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
    output.config(text=mf.get_prediction_str(predictions[0]))


def predictions_to_image(prediction):
    matplotlib.use("Agg")
    prediction_list = prediction[0].tolist()
    label_list = mf.get_labels()
    size_list = prediction_list
    sizes, labels = [], []
    for i, num in enumerate(prediction_list):
        if num >= .001:
            sizes.append(size_list[i])
            labels.append(label_list[i])
    plt.rcParams["figure.figsize"] = [3.50, 2.50]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels)
    #plt.show()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    im = Image.open(img_buf)
    pie_chart_img = ImageOps.contain(im, (256, 256))
    pie_chart_img = ImageTk.PhotoImage(pie_chart_img)

    #im.show(title="My Image")
    output.configure(image=pie_chart_img)
    output.image = pie_chart_img
    img_buf.close()


root = Tk()
root.title("Font Recognizer - Loading AI Framework")
root.geometry("900x400")

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
placeholder = ImageTk.PhotoImage(Image.open("utils/placeholder.png"))
preview_lbl = Label(main_screen, bd=2,  image=placeholder)
preview_lbl.grid(column=0, row=1, rowspan=2, sticky="S", padx=4, pady=6)
output_lbl = Label(main_screen, text="Font Certainty", height=1)
output_lbl.grid(column=1, row=1, sticky="S")
placeholder_short = ImageTk.PhotoImage(Image.open("utils/placeholder_short.png"))
output = Label(main_screen, bd=2,  image=placeholder_short)
output.grid(column=1, row=2, sticky="NSEW", padx=4, pady=6)

# dashboard tab widgets
dashboard_screen = Frame(notebook)
dashboard_screen.columnconfigure(0, minsize=100)
dashboard_screen.columnconfigure(1, minsize=400)
dashboard_screen.columnconfigure(2, minsize=100)

# notebook tabs
notebook.add(main_screen, text="Main")
notebook.add(dashboard_screen, text="Dashboard")

model_file = 'EfficientNetV2B1_model'
model = None


def callback(e):
    global model
    # your code here
    model = keras.models.load_model(model_file)
    root.title("Font Recognizer")
    root.unbind('<Visibility>')  # only call `callback` the first time `root` becomes visible


output.bind('<Visibility>', callback)

root.mainloop()
