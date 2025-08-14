""" Predicción de clases para nuevas imágenes """

import tensorflow as tf
import numpy as np

# Cargar modelo
model = tf.keras.models.load_model("models/cars_classificator.h5")

# Lista de clases — debe coincidir con la que da al cargar los datos 
clases = ['Camioneta', 'Pick-Up', 'Turismo']

# Ruta de la imagen a predecir
img_path = "samples/22r1982.jpg"

# Cargar y preprocesar la imagen
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))  # tamaño igual al del entrenamiento
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # agregar dimensión batch
img_array = img_array / 255.0  # normalizar igual que en entrenamiento

# Predecir la clase
pred = model.predict(img_array)
predicted_class = clases[np.argmax(pred)]

print(f"Imagen: {img_path}")
print(f"--------- Predicción: {predicted_class} ---------")
