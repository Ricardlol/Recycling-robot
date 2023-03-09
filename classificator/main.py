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


def grapics(number_epoch, val_acc, val_loss, acc, loss, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
    ax1.plot(number_epoch, loss, label='Train')
    ax1.plot(number_epoch, val_loss, label='Validation')
    ax1.set_title('Loss')
    ax1.set(xlabel='Epochs', ylabel='Loss')
    ax1.legend()
    ax2.plot(number_epoch, acc, label='Train')
    ax2.plot(number_epoch, val_acc, label='Validation')
    ax2.set_title('Accuracy')
    ax2.set(xlabel='Epochs', ylabel='Accuracy')
    ax2.legend()
    fig.suptitle(title, fontsize=16)


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

test_batches = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = preprocess_input) \
.flow_from_directory(directory=test_path, target_size=imageSize, classes=names_class, batch_size=batch_size, shuffle=False, class_mode='categorical')

'''
train_batches = augmentation(train_path, imageSize, names_class, batch_size)
test_batches = augmentation(test_path, imageSize, names_class, batch_size)
valid_batches = augmentation(validation_path, imageSize, names_class, batch_size)
'''
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

print(model.summary())

for layer in base_model.layers:
    layer.trainable = False;

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

first_model = model.fit(x=train_batches, steps_per_epoch=360, validation_steps=140, validation_data=valid_batches, epochs=30, verbose=1)

model.save("FirstModel.h5")
print("Save model")

## plot graphics
grapics(range(30), first_model.history['val_accuracy'], first_model.history['val_loss'], first_model.history['accuracy'], first_model.history['loss'], "DenseNet121- first_Model")

'''
for layer in model.layers[:313]:
    layer.trainable = False
for layer in model.layers[313:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

second_model = model.fit(
    x=train_batches,
    steps_per_epoch=420,
    validation_steps=140,
    validation_data=valid_batches,
    epochs=20,
    verbose=1
)
model.save("secondModel.h5")
print("Saved to drive")

grapics(20, second_model.history['val_accuracy'], second_model.history['val_loss'], second_model.history['accuracy'], second_model.history['loss'], "DenseNet121- Second Model")
'''