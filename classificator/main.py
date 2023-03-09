import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD

import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import re


# data augmentation
def augmentation(direct, size_image, classes, size_batch, mode='categorical'):
    aug = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        preprocessing_function=preprocess_input
    ).flow_from_directory(
        directory=direct,
        target_size=size_image,
        classes=classes,
        batch_size=size_batch,
        class_mode=mode
    )
    return aug


def plot_image_demo(img, title):
    plt.title(title)
    plt.imshow(img)
    plt.show()


def grapics_Model(model, title):
    plt.title('Cross Entropy Loss' + title)
    plt.plot(model.history['loss'], color='blue', label='train')
    plt.plot(model.history['val_loss'], color='orange', label='val')
    plt.legend()
    plt.show()

    plt.title('Classification Accuracy' + title)
    plt.plot(model.history['accuracy'], color='blue', label='train')
    plt.plot(model.history['val_accuracy'], color='orange', label='val')
    plt.legend()
    plt.show()


imageSize = (224, 224)
batch_size = 10
names_class = ['cardboard', 'glass', 'aluminium', 'organic', 'paper', 'plastic', 'trash']

train_path = './dataset/train'
test_path = './dataset/test'
validation_path = './dataset/valid'

train_batches = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40, horizontal_flip=True, width_shift_range=0.2,
        height_shift_range=0.2, zoom_range=0.2, fill_mode='nearest', preprocessing_function = preprocess_input) \
.flow_from_directory(directory=train_path, target_size=imageSize, classes=names_class, batch_size=batch_size, class_mode='categorical')

valid_batches = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = preprocess_input) \
.flow_from_directory(directory=validation_path, target_size=imageSize, classes=names_class, batch_size=batch_size, class_mode='categorical')


train_batches = augmentation(train_path, imageSize, names_class, batch_size)
valid_batches = augmentation(validation_path, imageSize, names_class, batch_size)

print("dataset Augmentet")

imgs, labels = next(train_batches)

# show images
plot_image_demo(imgs[0], "Example")

# create model
base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling='avg')

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

base_predic = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=base_predic)
print("model create")

'''
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

epoch = 30
steps = int(train_batches.samples / 15)
val_steps = int(valid_batches.samples / 15)

first_model = model.fit(train_batches, steps_per_epoch=steps, validation_steps=val_steps, validation_data=valid_batches, epochs=epoch, verbose=1)

model.save("FirstModel.h5")
print("Save model")

## plot graphics
grapics_Model( first_model, "first_Model")
'''

for layer in model.layers[:313]:
    layer.trainable = False
for layer in model.layers[313:]:
    layer.trainable = True

model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

epoch = 20
steps = int(train_batches.samples / 10)
val_steps = int(valid_batches.samples / 10)
second_model = model.fit(
    x=train_batches,
    steps_per_epoch=steps,
    validation_steps=val_steps,
    validation_data=valid_batches,
    epochs=epoch,
    verbose=1
)
model.save("secondModel.h5")
print("Saved to drive")

grapics_Model(second_model, "Second Model")
