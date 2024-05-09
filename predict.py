from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import config

model = load_model('waste_classification_model.h5')

def predict(image_path):
    img = image.load_img(image_path, target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    return prediction

if __name__ == "__main__":
    image_path = 'path_to_new_image.jpg'
    prediction = predict(image_path)
    print('Predicted class:', prediction)

