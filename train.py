# train.py

from data_loader import create_datasets
from model import build_model

import config
import os

def main():
    # Check data paths
    if not os.path.exists(config.TRAIN_DATA_PATH):
        raise Exception(f"Training data path does not exist: {config.TRAIN_DATA_PATH}")
    if not os.path.exists(config.TEST_DATA_PATH):
        raise Exception(f"Testing data path does not exist: {config.TEST_DATA_PATH}")
    # Proceed with data loading and model training
    train_dataset, validation_dataset = create_datasets()
    print("Train Dataset Classes:", train_dataset.class_names)
    print("Validation Dataset Classes:", validation_dataset.class_names)
    model = build_model()

    model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=validation_dataset
    )
    model.save('waste_classification_model.keras')

if __name__ == "__main__":
    main()
