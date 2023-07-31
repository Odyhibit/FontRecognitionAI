from tensorflow import keras as keras

train_ds = keras.utils.image_dataset_from_directory(
    directory='train_data/',
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224)
)

test_ds = keras.utils.image_dataset_from_directory(
    directory='test_data/',
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224)
)

model = keras.applications.MobileNetV2(weights=None, classes=5, input_shape=(224, 224, 3))

# EfficientNetV2B1(weights=None, classes=5, input_shape=(224, 224, 3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=10, validation_data=test_ds)

model.save("mobile_plus")

