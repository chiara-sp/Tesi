# data_loader.py

import tensorflow as tf
import config

def load_and_preprocess_image(path):
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [config.IMAGE_HEIGHT, config.IMAGE_WIDTH])
        image = image / 255.0  # Normalize to [0, 1]
        return image
    except tf.errors.NotFoundError:
        print("File not found:", path)
        return None

def prepare_dataset(data_directory, batch_size):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_directory,
        label_mode='categorical',  # For multi-class classification
        image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        batch_size=batch_size,
        shuffle=True
    )
    return dataset

def create_datasets():
    train_dataset = prepare_dataset(config.TRAIN_DATA_PATH, config.BATCH_SIZE)
    validation_dataset = prepare_dataset(config.TEST_DATA_PATH, config.BATCH_SIZE)
    return train_dataset, validation_dataset
