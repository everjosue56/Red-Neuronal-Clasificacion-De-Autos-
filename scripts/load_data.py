""" Cargar y preprocesar dataset de imágenes """

import tensorflow as tf

# Tamaño al que redimensionaras las imagenes
img_height, img_width = 128, 128
batch_size = 32

# Directorios
train_dir = 'dataset/train'
valid_dir = 'dataset/valid'
test_dir = 'dataset/test'

# Cargar datos de entrenamiento y validación
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen  = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

valid_data = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Mostrar las clases
print("Clases:", train_data.class_indices)
