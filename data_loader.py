# data_loader.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config

def create_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DATA_PATH,
        target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        config.TEST_DATA_PATH,
        target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical'
    )

    return train_generator, validation_generator

