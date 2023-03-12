import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.densenet import preprocess_input

import cv2
import matplotlib.pyplot as plt
import numpy as np
import itertools
import re

class modelsDenseNet121:
    def __init__(self, epoch, steps, val_steps, train_batches, valid_batches, test ,optimization=False):
        self.base_model = ''
        self.model = ''
        self.base_predic = ''
        self.predic = ''
        self.confusion = ""
        self.epoch = epoch
        self.steps = steps
        self.val_steps = val_steps
        self.train_batches = train_batches
        self.valid_batches = valid_batches
        self.testData = test
        self.optimization = optimization

    # Create model Denset121
    def baseModelDenseNet121(self):
        self.base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling='avg')

        x = self.base_model.output
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

        self.base_predic = Dense(7, activation='softmax')(x)

        self.model = Model(inputs=self.base_model.input, outputs=self.base_predic)

        print("model create")

    def compileModel(self):
        if self.optimization:
            self.model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy',
                               metrics=['accuracy'])
        else:
            self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self):
        self.model.fit(self.train_batches, steps_per_epoch=self.steps, validation_steps=self.val_steps,
                       validation_data=self.valid_batches, epochs=self.epoch, verbose=1)

    def predictions(self):
        self.predic = self.model.predict(x=self.testData, verbose = 1, steps = self.val_steps)

    def plotGraphics(self, title):
        plt.title('Cross Entropy Loss' + title)
        plt.plot(self.model.history['loss'], color='blue', label='train')
        plt.plot(self.model.history['val_loss'], color='orange', label='val')
        plt.legend()
        plt.show()

        plt.title('Classification Accuracy' + title)
        plt.plot(self.model.history['accuracy'], color='blue', label='train')
        plt.plot(self.model.history['val_accuracy'], color='orange', label='val')
        plt.legend()
        plt.show()

    def createConfuMat(self):
        self.confusion = confusion_matrix(y_true=self.testData.classes, y_pred=np.argmax(self.predic, axis=-1))

    def plotConfusionMatrix(self, classes, title="Confusion Matrix", normalize=False, cmap = plt.cm.Blues):
        self.createConfuMat()
        plt.imshow(self.confusion, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            self.confusion = self.confusion.astype('float') / self.confusion.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(self.confusion)

        thresh = self.confusion.max() / 2.
        for i, j in itertools.product(range(self.confusion.shape[0]), range(self.confusion.shape[1])):
            plt.text(j, i, self.confusion[i, j],
                     horizontalalignment="center",
                     color="white" if self.confusion[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def save(self, name):
        self.model.save("./models/" + name + ".h5")
        print("Save model")

    def loadModel(self, name):
        self.model = load_model("./models/" + name + ".h5")
        print("load model")

    def layerFalse(self, num=0):
        if num == 0:
            for layer in self.base_model.layers:
                layer.trainable = False
        else:
            for layer in self.model.layers[:num]:
                layer.trainable = False
            for layer in self.model.layers[num:]:
                layer.trainable = True

    def showIndexesTest(self):
        print(self.testData.class_indices)

    def testing(self, classes):
        img_test = "./test/organic/organic (31).jpg"
        img = tf.image.decode_jpeg(tf.io.read_file(img_test), channels=3)
        dimension = np.expend_dims(img, axis=0)
        imageResize = tf.image.resize_with_pad(dimension, 224,224, method="bilinear", antialias=True)
        imageProces = preprocess_input(imageResize)

        img = cv2.imread(img_test, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        plt.imshow(img)
        plt.show()

        results = self.model.predict(imageProces)
        print('Class: ' + classes[np.argmax(results)] + ' ' + str(round(results[0, np.argmax(results)] * 100, 2)) + '%')


