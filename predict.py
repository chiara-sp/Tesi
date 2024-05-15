# predict.py

import tensorflow as tf
import numpy as np
import config
from data_loader import load_and_preprocess_image  # Import the shared function

model = tf.keras.models.load_model('waste_classification_model.keras')

def predict(image_path):
    img = load_and_preprocess_image(image_path)
    if img is None:
        return "Image could not be loaded or processed."

    img_array = tf.expand_dims(img, axis=0)  # Add a batch dimension
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]

    class_names= ['cardboard', 'glass','metal','paper','plastic','trash']
    predicted_class_name=class_names[predicted_class_index]
    return predicted_class_name, confidence

if __name__ == "__main__":
    image_path = '/Users/chiaraspirito/Desktop/dataset-resized/train/trash/trash1.jpg'  # Example image path
    predicted_class_index, confidence = predict(image_path)
    print(f'Predicted class index: {predicted_class_index} with confidence {confidence:.2f}')


