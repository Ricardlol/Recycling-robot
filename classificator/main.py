import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
import classModels


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

print("dataset Augmented")

'''
epoch = 30
steps = int(train_batches.samples / 15)
val_steps = int(valid_batches.samples / 15)

modelfirst = classModels.modelsDenseNet121(epoch, steps, val_steps, train_batches, valid_batches, test)

modelfirst.layerFalse()

modelfirst.compileModel()

modelfirst.fit()

modelfirst.fit()

modelfirst.save("FirstModel")

modelfirst.plotGraphics("First model")
'''

'''
epoch = 20
steps = int(train_batches.samples / 10)
val_steps = int(valid_batches.samples / 10)

secondModel = classModels.modelsDenseNet121(epoch, steps, val_steps, train_batches, valid_batches, test, True)
secondModel.layerFalse(313)

secondModel.compileModel()

secondModel.fit()

secondModel.save

secondModel.plotGraphics("SeconModel")
'''
epoch = 20
steps = int(train_batches.samples / 10)
val_steps = int(valid_batches.samples / 10)

secondModel = classModels.modelsDenseNet121(epoch, steps, val_steps, train_batches, valid_batches, test, True)

secondModel.loadModel("secondModel")

secondModel.showIndexesTest()
secondModel.predictions()
secondModel.plotConfusionMatrix(names_class, "Confusion Matrix (best model)")

secondModel.testing(names_class)