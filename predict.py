# predict.py

import tensorflow as tf
import numpy as np
import config
import os
from data_loader import load_and_preprocess_image  # Import the shared function

model = tf.keras.models.load_model('waste_classification_model.keras')


def predict_all_images(data_directory):
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    predictions = []

    for class_name in class_names:
        class_path = os.path.join(data_directory, class_name)
        image_files = [os.path.join(class_path, fname) for fname in os.listdir(class_path)]

        for image_file in image_files:
            img = load_and_preprocess_image(image_file)
            if img is not None:
                img_array = tf.expand_dims(img, axis=0)  # Add a batch dimension
                prediction = model.predict(img_array)[0]
                predicted_class_index = np.argmax(prediction)
                confidence = prediction[predicted_class_index]
                predicted_class_name = class_names[predicted_class_index]
                predictions.append((image_file, predicted_class_name, confidence))

    return predictions


if __name__ == "__main__":
    data_directory = '/Users/chiaraspirito/Desktop/dataset-resized/train'
    all_predictions = predict_all_images(data_directory)

    for image_file, predicted_class, confidence in all_predictions:
        print(f'Image: {image_file}, Predicted class: {predicted_class}, Confidence: {confidence:.2f}')


