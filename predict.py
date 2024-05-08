
from tensorflow.keras.preprocessing import image
import numpy as np
import config
from model import build_model

model = build_model()
model.load_weights('path_to_saved_model.h5')  # Update this path

def predict(image_path):
    img = image.load_img(image_path, target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    return class_names[np.argmax(prediction)]

# Example usage
if __name__ == "__main__":
    image_path = 'path_to_new_image.jpg'
    result = predict(image_path)
    print('Predicted class:', result)
