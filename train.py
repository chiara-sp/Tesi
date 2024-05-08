from data_loader import load_data
from model import build_model
import config

train_generator, test_generator = load_data()
model = build_model()

model.fit(
    train_generator,
    steps_per_epoch=100,  # Adjust based on your dataset
    epochs=config.EPOCHS,
    validation_data=test_generator,
    validation_steps=50  # Adjust based on your dataset
)
