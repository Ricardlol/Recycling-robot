import tensorflow as tf
from tensorflow import keras

from keras import regularizers
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.densenet import preprocess_input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import cv2
import matplotlib.pyplot as plt
import numpy as np
import itertools
from PIL import Image
import re

class modelsDenseNet121:
    def __init__(self, epoch, steps, val_steps, train_batches, valid_batches, test, test_prepro, optimization=False):
        self.base_model = ''
        self.model = ''
        self.base_predic = ''
        self.predic = ''
        self.confusion = ""
        self.historic = ''
        self.callbackList = ''
        self.model_prepro = ""
        self.epoch = epoch
        self.steps = steps
        self.val_steps = val_steps
        self.train_batches = train_batches
        self.valid_batches = valid_batches
        self.testData = test
        self.optimization = optimization
        self.test_nopre = test_prepro

    def __init__(self, nameModel):
        self.loadModel(nameModel)

    # Create model Denset121
    def baseModelDenseNet121(self):
        self.base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling='avg')

        x = self.base_model.output
        x = Flatten()(x)
        x = Dense(256, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01), activation='relu')(x)
        x = Dropout(0.5)(x)
        self.base_predic = Dense(4, activation='softmax')(x)

        self.model = Model(inputs=self.base_model.input, outputs=self.base_predic)

        print("model create")

    def compileModel(self):
        if self.optimization:
            self.model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy',
                               metrics=['accuracy', 'mse'])
            stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
            reduceLr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_delta=1e-6)
            self.callbackList = [stop, reduceLr]
            print("Optimization")
        else:
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse'])
            stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
            reduceLr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4)
            self.callbackList = [stop, reduceLr]
    def fit(self):
         self.historic=self.model.fit(
             self.train_batches,
             steps_per_epoch=self.steps,
             validation_steps=self.val_steps,
             validation_data=self.valid_batches,
             epochs=self.epoch,
             callbacks=self.callbackList,
             verbose=1
         )

    def predictions(self, preproc=False):
        if preproc:
            self.predic = self.model.predict(x=self.test_nopre, verbose=1, steps=self.val_steps)
        else:
            self.predic = self.model.predict(x=self.testData, verbose=1, steps=self.val_steps)

    def evaluateModel(self):
        self.model.evaluate(x=self.testData, verbose=1, steps=self.val_steps)
    def plotGraphics(self, title):
        plt.title('Training Loss vs Validarion Loss ' + title)
        plt.plot(self.historic.history['loss'], color='blue', label='train')
        plt.plot(self.historic.history['val_loss'], color='orange', label='val')
        plt.xlabel("Num of Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.title('Training Accuracy vs Validation Accuracy ' + title)
        plt.plot(self.historic.history['accuracy'], color='blue', label='train')
        plt.plot(self.historic.history['val_accuracy'], color='orange', label='val')
        plt.xlabel("Num of Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        plt.title('Training MSE vs Validarion MSE ' + title)
        plt.plot(self.historic.history['mse'], color='blue', label='train')
        plt.plot(self.historic.history['val_mse'], color='orange', label='val')
        plt.xlabel("Num of Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.show()

    def createConfuMat(self, prepro=False):
        if (prepro):
            self.confusion = confusion_matrix(y_true=self.test_nopre.classes, y_pred=np.argmax(self.predic, axis=-1))
        else:
            self.confusion = confusion_matrix(y_true=self.testData.classes, y_pred=np.argmax(self.predic, axis=-1))
        print("complete predictions")
    def plotConfusionMatrix(self, classes, prepro = False, title="Confusion Matrix", normalize=False, cmap=plt.cm.Blues):
        self.createConfuMat(prepro)
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
        self.model_prepro = load_model("./models/" + name + ".h5")
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
        img_test = "./dataset/test/organic/organic (31).jpg"
        img = tf.image.decode_jpeg(tf.io.read_file(img_test), channels=3)
        dimension = np.expand_dims(img, axis=0)
        imageResize = tf.image.resize_with_pad(dimension, 224,224, method="bilinear", antialias=True)
        imageProces = preprocess_input(imageResize)

        results = self.model.predict(imageProces)
        print('Class: ' + classes[np.argmax(results)] + ' ' + str(round(results[0, np.argmax(results)] * 100, 2)) + '%')

        img = cv2.imread(img_test, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        plt.imshow(img)
        plt.title("Class: " + classes[np.argmax(results)] + ' ' + str(round(results[0, np.argmax(results)] * 100, 2)) + '%')
        plt.show()

    def myImages(self, classes, nameFile):
        img = Image.open(nameFile)

        imgResize = img.resize((224, 224), Image.BILINEAR)

        dimension = np.array(imgResize)
        imgDimension = np.expand_dims(dimension, axis=0)

        results = self.model_prepro.predict(imgDimension)

        print('Class: ' + classes[np.argmax(results)] + ' ' + str(round(results[0, np.argmax(results)] * 100, 2)) + '%')

        img = cv2.imread(nameFile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title("Class: " + classes[np.argmax(results)] + ' ' + str(round(results[0, np.argmax(results)] * 100, 2)) + '%')
        plt.show()

    def classifyImage(self, classes, nameFile):
        img = Image.open(nameFile)

        imgResize = img.resize((224, 224), Image.BILINEAR)

        dimension = np.array(imgResize)
        imgDimension = np.expand_dims(dimension, axis=0)

        results = self.model_prepro.predict(imgDimension)
        classResult = 0

        if classes[np.argmax(results)] == "yellow":
             classResult = 0
        elif classes[np.argmax(results)] == "blue":
             classResult = 1
        elif classes[np.argmax(results)] == "green":
            classResult = 2
        else:
            classResult = 3

        return classResult


    def preprocesing(self, classes):
        i = Input([None, None, 3], dtype=tf.uint8)
        x = tf.cast(i, tf.float32)
        x = preprocess_input(x)
        x = self.model(x)
        self.model_prepro = Model(inputs=[i], outputs=[x])

        self.model_prepro.compile(optimizer=SGD(learning_rate=0.00001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model_prepro.save("./models/modelprepro.h5")
        self.predic = self.model_prepro.predict(x=self.test_nopre, verbose=1, steps=self.steps)

        self.predic = np.round(self.predic)

        self.plotConfusionMatrix(classes, True, "Confusion Matrix (preproces)")
