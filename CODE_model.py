import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from shutil import copyfile
import matplotlib.pyplot as plt

data_dir = "C:\\Users\\ADMIN\\Desktop\\binsmort\\codes\\data"
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "validation")

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    batch_size=128,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    valid_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              metrics=['accuracy'])

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95 and logs.get('val_accuracy') > 0.80:
            print("\n Accuracy is high so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()


history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=8,
    verbose=2,
    callbacks=[callbacks]
)


model.save('model_01.h5')
print("Model saved as model_01.h5")