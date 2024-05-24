import os
import pandas as pd
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import config
from PIL import Image


class DatasetWithClassNames:
    def __init__(self, dataset, class_names):
        self.dataset = dataset
        self.class_names = class_names

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


def get_filenames_and_labels(directory):
    filenames = []
    labels = []
    class_names = sorted(os.listdir(directory)) [1:] # Make sure directory only contains valid class folders
    label_dict = {name: index for index, name in enumerate(class_names)}

    for label_name in class_names:
        class_dir = os.path.join(directory, label_name)
        class_files = [os.path.join(class_dir, name) for name in os.listdir(class_dir) if
                       name.endswith(('.png', '.jpg', '.jpeg'))]  # Make sure to filter only image files
        filenames.extend(class_files)
        labels.extend([label_dict[label_name]] * len(class_files))

    return filenames, labels, class_names

# Function for data augmentation
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def create_datasets_for_fold(train_files, train_labels, val_files, val_labels, class_names):
    # Creating file datasets for both training and validation
    train_data = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_data = tf.data.Dataset.from_tensor_slices((val_files, val_labels))

    # Apply the original function to prepare datasets
    train_dataset = train_data.map(lambda x, y: (load_and_preprocess_image(x), tf.one_hot(y, depth=len(class_names))))
    validation_dataset = val_data.map(lambda x, y: (load_and_preprocess_image(x), tf.one_hot(y, depth=len(class_names))))

    #Create augmented datasets
    augmented_datasets = [train_dataset.map(augment_image) for _ in range(9)]  # Create 9 augmented datasets for train
    augmented_datasets_val = [validation_dataset.map(augment_image) for _ in range(3)]

    # Concatenate original and augmented datasets
    full_train_dataset = train_dataset.concatenate(augmented_datasets[0])
    for aug_dataset in augmented_datasets[1:]:
        full_train_dataset = full_train_dataset.concatenate(aug_dataset)
    full_val_dataset = validation_dataset.concatenate(augmented_datasets_val[0])
    for aug_dataset in augmented_datasets_val[1:]:
        full_val_dataset = full_val_dataset.concatenate(aug_dataset)

    # Batch the datasets
    train_dataset = full_train_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    validation_dataset = full_val_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset
''''''
def printIMG(train):
    # Iterate over each trash type (folder) to display images
    # Set up subplots
    for i, garbage in enumerate(train.filenames[:10]):
        # Select the first 10 images

        with Image.open(garbage) as img:
            plt.imshow(img)

        plt.tight_layout()
        plt.title(garbage)
        plt.show()
