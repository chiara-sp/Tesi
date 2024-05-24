# train.py

from data_loader import create_datasets
from model import build_model
from model import ResNet18
import config
import os

def main():
    n_splits = 5  # Number of folds for cross-validation
    fold_results = []

    for fold, (train_dataset_with_names, validation_dataset_with_names) in enumerate(create_datasets(n_splits=n_splits),
                                                                                     start=1):
        print(f"Training fold {fold}/{n_splits}...")

        train_dataset = train_dataset_with_names.dataset
        validation_dataset = validation_dataset_with_names.dataset

        train_class_names = train_dataset_with_names.class_names
        validation_class_names = validation_dataset_with_names.class_names
        print("Train Dataset Classes:", train_class_names)
        print("Validation Dataset Classes:", validation_class_names)

        # Build and compile your model
        model = build_model()  # Make sure you have a build_model function or similar
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(
            train_dataset,
            epochs=config.EPOCHS,
            validation_data=validation_dataset
        )

        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(validation_dataset)
        print(f"Fold {fold} - Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")

        # Save results for this fold
        fold_results.append({
            'fold': fold,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })

    # Save the final model after the last fold
    model.save('waste_classification_model.keras')

    # Print summary of results
    for result in fold_results:
        print(
            f"Fold {result['fold']} - Validation Accuracy: {result['val_accuracy']:.4f}, Validation Loss: {result['val_loss']:.4f}")


if __name__ == "__main__":
    main()
