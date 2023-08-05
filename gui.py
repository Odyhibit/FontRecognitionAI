import tkinter.filedialog
from tkinter import *
from tkinter import ttk
import main
from tensorflow import keras as keras
from PIL import Image, ImageTk, ImageOps


def pick_cover(event: Event = Event()):
    global model
    file_name = tkinter.filedialog.askopenfilename(initialdir="./test_images", filetypes=[("image files", ".jpg")])
    if not file_name:
        return
    image_path_str.set(file_name)
    test_img = main.load_image(file_name)
    preview_img = Image.open(file_name)
    preview_img = ImageTk.PhotoImage(preview_img)
    preview_lbl.configure(image=preview_img)
    preview_lbl.image = preview_img
    predictions = main.predict_from_image(test_img, model)
    output.config(text=main.get_predictions(predictions[0]))


'''
splash_root = Tk()
splash_root.geometry("700x200")
Label(splash_root, text="Loading AI framework").pack(pady= 20)
splash_root.mainloop()"
'''

root = Tk()
root.title("Font Recognizer")
root.geometry("700x400")

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
placeholder = ImageTk.PhotoImage(Image.open("placeholder.png"))
preview_lbl = Label(main_screen, bd=2, relief="groove", image=placeholder)
preview_lbl.grid(column=0, row=1, rowspan=2, sticky="S", padx=4, pady=6)
output_lbl = Label(main_screen, text="Font   Certainty", height=1)
output_lbl.grid(column=1, row=1, sticky="S")
output = Label(main_screen, bd=2, relief="groove", width=16, height=14)
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
    root.unbind('<Visibility>')  # only call `callback` the first time `root` becomes visible


root.bind('<Visibility>', callback)  # call `callback` whenever `root` becomes visible

root.mainloop()
