# train.py

from data_loader import create_generators
from model import build_model
import config

def main():
    train_generator, validation_generator = create_generators()
    model = build_model()

    model.fit(
        train_generator,
        steps_per_epoch=100,  # Adjust based on your dataset size
        epochs=config.EPOCHS,
        validation_data=validation_generator,
        validation_steps=50  # Adjust based on validation dataset size
    )

    model.save('waste_classification_model.h5')  # Saving the trained model

if __name__ == "__main__":
    main()
