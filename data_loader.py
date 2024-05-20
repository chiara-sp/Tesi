# data_loader.py
import os
import pandas as pd
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import config
from PIL import Image
from sklearn.model_selection import train_test_split


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



def data_Augmentation(train, test):
    # Slight Augmentation settings for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # Normalize pixel values to [0,1]
        rotation_range=45,  # Randomly rotate the images by up to 45 degrees
        width_shift_range=0.15,  # Randomly shift images horizontally by up to 15% of the width
        height_shift_range=0.15,  # Randomly shift images vertically by up to 15% of the height
        zoom_range=0.15,  # Randomly zoom in or out by up to 15%
        horizontal_flip=True,  # Randomly flip images horizontally
        vertical_flip=True,  # Randomly flip images vertically
        shear_range=0.05,  # Apply slight shear transformations
        brightness_range=[0.9, 1.1],  # Vary brightness between 90% to 110% of original
        channel_shift_range=10,  # Randomly shift channels (can change colors of images slightly but less aggressively)
        fill_mode='nearest'  # Fill in missing pixels using the nearest filled value
    )
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    # Using flow_from_dataframe to generate batches
    # Generate training batches from the training dataframe
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train,  # DataFrame containing training data
        x_col="filepath",  # Column with paths to image files
        y_col="label",  # Column with image labels
        target_size=(384, 384),  # Resize all images to size of 384x384
        batch_size=32,  # Number of images per batch
        class_mode='categorical',  # One-hot encode labels
        seed=42,  # Seed for random number generator to ensure reproducibility
        shuffle=False  # Data is not shuffled; order retained from DataFrame
    )

    # Generate validation batches from the validation dataframe
    test_gen = val_datagen.flow_from_dataframe(
        dataframe=test,  # DataFrame containing validation data
        x_col="filepath",  # Column with paths to image files
        y_col="label",  # Column with image labels
        target_size=(384, 384),  # Resize all images to size of 384x384
        batch_size=32,  # Number of images per batch
        class_mode='categorical',  # One-hot encode labels
        seed=42,  # Seed for random number generator to ensure reproducibility
        shuffle=False  # Data is not shuffled; order retained from DataFrame
    )
    return train_gen, test_gen

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

''''''
def printIMG(train, test):
    # Iterate over each trash type (folder) to display images
    # Set up subplots
    for i, garbage in enumerate(train.filenames[:10]):
        # Select the first 10 images

        with Image.open(garbage) as img:
            plt.imshow(img)

        plt.tight_layout()
        plt.title(garbage)
        plt.show()

def prepare_dataset_v2():
    # Initialize an empty list to store image file paths and their respective labels
    data = []
    label = os.listdir(config.TRAIN_DATA_PATH)
    label.sort()
    # Loop through each garbage type and collect its images' file paths
    for garbage_type in label[1:7]:
        for file in os.listdir(os.path.join(config.TRAIN_DATA_PATH, garbage_type)):
            # Append the image file path and its trash type (as a label) to the data list
            data.append((os.path.join(config.TRAIN_DATA_PATH, garbage_type, file), garbage_type))

    # Convert the collected data into a DataFrame
    df = pd.DataFrame(data, columns=['filepath', 'label'])
    return df
    # Display the first few entries of the DataFrame
    #df.head()