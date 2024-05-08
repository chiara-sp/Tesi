from data_loader import load_data
from model import build_model

_, test_generator = load_data()
model = build_model()
model.load_weights('path_to_saved_model.h5')  # Update this path

test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)
