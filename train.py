# train.py

from data_loader import create_datasets
from data_loader import create_datasets_v2
from model import build_model
from model import ResNet18
import config
import os

def main():
    # Check data paths
    if not os.path.exists(config.TRAIN_DATA_PATH):
        raise Exception(f"Training data path does not exist: {config.TRAIN_DATA_PATH}")
    if not os.path.exists(config.TEST_DATA_PATH):
        raise Exception(f"Testing data path does not exist: {config.TEST_DATA_PATH}")
    # Proceed with data loading and model training
    #train_dataset, validation_dataset = create_datasets_v2()

    train_dataset_with_names, validation_dataset_with_names = create_datasets()
    train_dataset= train_dataset_with_names.dataset
    validation_dataset= validation_dataset_with_names.dataset

    train_class_names= train_dataset_with_names.class_names
    validation_class_names= validation_dataset_with_names.class_names
    print("Train Dataset Classes:", train_class_names)
    print("Validation Dataset Classes:", validation_class_names)
    ''''''
    model = build_model()
    model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=validation_dataset
    )
    model.save('waste_classification_model.keras')
''''''
if __name__ == "__main__":
    main()
