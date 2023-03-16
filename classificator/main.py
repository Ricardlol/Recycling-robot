import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
import classModels


imageSize = (224, 224)
batch_size = 32
names_class = ['blue', 'green', 'organic', 'yellow']

train_path = './dataset/train'
test_path = './dataset/test'
validation_path = './dataset/valid'

train_batches = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=5,
        rescale=1./255,
        horizontal_flip=True,
        zoom_range=0.2,
).flow_from_directory(
        directory=train_path,
        target_size=imageSize,
        classes=names_class,
        batch_size=batch_size,
        class_mode='categorical'
)

valid_batches = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input
).flow_from_directory(
        directory=validation_path,
        target_size=imageSize,
        classes=names_class,
        batch_size=batch_size,
        class_mode='categorical'
)

test_batches = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
).flow_from_directory(
        directory=test_path,
        target_size=imageSize,
        classes=names_class,
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical'
)
test_batches_nopre = tf.keras.preprocessing.image.ImageDataGenerator(
        dtype='uint8'
).flow_from_directory(
        directory=test_path,
        target_size=imageSize,
        classes=names_class,
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical'
)

print("dataset Augmented")

epoch = 20
steps = int(train_batches.samples / 35)
val_steps = int(valid_batches.samples / 35)

secondModel = classModels.modelsDenseNet121(epoch, steps, val_steps, train_batches, valid_batches, test_batches, test_batches_nopre, True)

#secondModel.baseModelDenseNet121()

#secondModel.layerFalse(313)

#secondModel.compileModel()

#secondModel.fit()

#secondModel.save(" SecondModel")

#secondModel.plotGraphics("SeconModel")

secondModel.loadModel(" SecondModel")

#secondModel.evaluateModel()

#secondModel.predictions()

#secondModel.plotConfusionMatrix(names_class, "Confusion Matrix (best model)")

#secondModel = classModels.modelsDenseNet121(epoch, steps, val_steps, train_batches, valid_batches, test_batches, True)

#secondModel.loadModel("model(0_85)")

#secondModel.evaluateModel()

#secondModel.showIndexesTest()
#secondModel.predictions()
#secondModel.plotConfusionMatrix(names_class, "Confusion Matrix (best model)")

#secondModel.testing(names_class)
secondModel.preprocesing(names_class)
secondModel.myImages(names_class, "./dataset/myImages/green5.JPG")
secondModel.myImages(names_class, "./dataset/myImages/yellow.JPG")