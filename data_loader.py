from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config

def load_data():
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DATA_PATH,
        target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        config.TEST_DATA_PATH,
        target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical')

    return train_generator, test_generator
