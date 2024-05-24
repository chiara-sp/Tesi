# train.py

from data_loader import create_datasets_for_fold
from model import build_model
from model import ResNet18
import config
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedKFold
from data_loader import get_filenames_and_labels

def perform_cross_validation(filenames, labels, class_names, num_folds=2):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_no = 1
    results = []

    for train_index, val_index in skf.split(filenames, labels):
        print(f"Training on fold {fold_no}...")

        train_files, val_files = filenames[train_index], filenames[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        # Create datasets for this fold
        train_dataset, validation_dataset = create_datasets_for_fold(train_files, train_labels, val_files, val_labels, class_names)

        # Build a new model for this fold
        model = build_model()

        # Train the model
        history = model.fit(
            train_dataset,
            epochs=config.EPOCHS,
            validation_data=validation_dataset
        )

        # Evaluate the model on the validation dataset
        loss, accuracy = model.evaluate(validation_dataset)
        print(f"Fold {fold_no} - Loss: {loss}, Accuracy: {accuracy}")

        results.append((loss, accuracy))
        fold_no += 1

    return results
def main():
    # Check data paths
    if not os.path.exists(config.TRAIN_DATA_PATH):
        raise Exception(f"Training data path does not exist: {config.TRAIN_DATA_PATH}")

    filenames, labels, class_names = get_filenames_and_labels(config.TRAIN_DATA_PATH)
    filenames = np.array(filenames)
    labels = np.array(labels)

    # Perform cross-validation
    results = perform_cross_validation(filenames, labels, class_names, num_folds=5)

    # Print the average performance across all folds
    avg_loss = np.mean([result[0] for result in results])
    avg_accuracy = np.mean([result[1] for result in results])
    print(f"Average Loss: {avg_loss}, Average Accuracy: {avg_accuracy}")

if __name__ == "__main__":
    main()
