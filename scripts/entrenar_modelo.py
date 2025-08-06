import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuracion
img_height, img_width = 128, 128
batch_size = 32

train_dir = 'dataset/train'
valid_dir = 'dataset/valid'
test_dir = 'dataset/test'   

# Cargar datos
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

valid_data = datagen.flow_from_directory(
    valid_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False 
)

# Construccion del modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # Cambia según número de clases
])

# Compilación del modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento
epochs = 10
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs
)

# Guardar el modelo entrenado
os.makedirs("modelos", exist_ok=True)
model.save('modelos/modelo_carros.h5')

# Graficar precisión y pérdida
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.legend()

os.makedirs("resultados", exist_ok=True)
plt.savefig('resultados/graficas_entrenamiento.png')
plt.show()

# Evaluación con el conjunto de prueba
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'\nPérdida en test: {test_loss:.4f}')
print(f'Precisión en test: {test_accuracy:.4f}')

# Predicción sobre todo el conjunto test
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes  # Etiquetas verdaderas
class_labels = list(train_data.class_indices.keys())

# Mostrar primeras 6 imágenes con predicción y etiqueta real
plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    img = test_generator[i][0][0]  # imagen i del batch i (ajustado para mostrar una imagen)
    plt.imshow(img)
    plt.title(f"Real: {class_labels[true_classes[i]]}\nPredicho: {class_labels[predicted_classes[i]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

print("-------- Modelo guardado como modelos/modelo_carros.h5  ---------")
