
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = load_model('D:/animal_classifier_model.h5')

# Load and preprocess the image
img = image.load_img('D:/cow.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant', 'Giraffe',
               'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']
predicted_class = class_names[np.argmax(pred)]
print("Predicted Animal:", predicted_class)
