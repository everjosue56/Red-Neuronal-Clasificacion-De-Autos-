""" Construir y cargar el modelo """

import tensorflow as tf

# Cargar el modelo previamente guardado
modelo_cargado = tf.keras.models.load_model("models/cars_classificator.h5")

# Confirmar carga
print("------- Modelo cargado correctamente. ---------")
