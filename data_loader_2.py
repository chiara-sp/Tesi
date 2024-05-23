import os
import pandas as pd
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import config
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

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
def data_Augmentation(train_df, test_df):
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

    # Generate training batches from the training dataframe
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,  # DataFrame containing training data
        x_col="filepath",  # Column with paths to image files
        y_col="label",  # Column with image labels
        target_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        # Resize all images to size of config.IMAGE_HEIGHT x config.IMAGE_WIDTH
        batch_size=config.BATCH_SIZE,  # Number of images per batch
        class_mode='categorical',  # One-hot encode labels
        seed=42,  # Seed for random number generator to ensure reproducibility
        shuffle=True  # Data is shuffled
    )

    # Generate validation batches from the validation dataframe
    test_gen = val_datagen.flow_from_dataframe(
        dataframe=test_df,  # DataFrame containing validation data
        x_col="filepath",  # Column with paths to image files
        y_col="label",  # Column with image labels
        target_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        # Resize all images to size of config.IMAGE_HEIGHT x config.IMAGE_WIDTH
        batch_size=config.BATCH_SIZE,  # Number of images per batch
        class_mode='categorical',  # One-hot encode labels
        seed=42,  # Seed for random number generator to ensure reproducibility
        shuffle=False  # Data is not shuffled; order retained from DataFrame
    )
    print(f"Number of batches in train_generator: {len(train_gen)}")
    print(f"Number of batches in val_generator: {len(test_gen)}")
    return train_gen, test_gen

def create_datasets_v2():
    train_df = prepare_dataset_v2()
    # Split with stratification
    train_dataset, validation_dataset = train_test_split(train_df, test_size=0.3, random_state=42, stratify=train_df['label'])

    # Print the number of images in each set
    print(f"Number of images in the training set: {len(train_dataset)}")
    print(f"Number of images in the validation set: {len(validation_dataset)}")

    # 1. Class distribution in the entire dataset
    overall_distribution = train_df['label'].value_counts(normalize=True) * 100

    # 2. Class distribution in the training set
    train_distribution = train_dataset['label'].value_counts(normalize=True) * 100

    # 3. Class distribution in the validation set
    val_distribution = validation_dataset['label'].value_counts(normalize=True) * 100

    print("Class distribution in the entire dataset:\n")
    print(overall_distribution.round(2))
    print('-' * 40)

    print("\nClass distribution in the training set:\n")
    print(train_distribution.round(2))
    print('-' * 40)

    print("\nClass distribution in the validation set:\n")
    print(val_distribution.round(2))

    #train_dataset, validation_dataset= data_Augmentation(train_dataset, validation_dataset)
    return train_dataset, validation_dataset

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

    # Display the first few entries of the DataFrame
    #df.head()
    return df