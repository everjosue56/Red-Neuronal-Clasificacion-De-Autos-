#  Red Neuronal para Clasificación de Autos

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API%20de%20TensorFlow-D00000?logo=keras)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Usado-013243?logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualización-11557C?logo=matplotlib)](https://matplotlib.org/)

---

## ¿De qué trata nuestra red neuronal?

Este proyecto implementa una red neuronal utilizando TensorFlow para resolver el problema de clasificación de autos a partir de imágenes. La clasificación se divide en tres categorías: **Turismo, Pick-Up y Camioneta**.

---

##  Funcionalidades principales

- Analiza imágenes enviadas por el usuario.
- Determina a qué categoría pertenece el vehículo.
- Clasificación precisa con base en entrenamiento supervisado.

---

##  Dataset

En esta ocasión, hemos utilizado un dataset especialmente diseñado para la clasificación de imágenes de autos.  

 Enlace directo al dataset:
```bash
https://www.kaggle.com/datasets/ademboukhris/cars-body-type-cropped
```

## Arquitectura del Modelo

El modelo está compuesto por diferentes capas de una red neuronal convolucional (CNN), cada una con una función específica en el proceso de clasificación de imágenes:

- **`Conv2D` y `MaxPooling2D`**: Detectan y extraen patrones visuales importantes como bordes, formas y texturas en las imágenes.  
- **`Flatten` y `Dense`**: Transforman las características extraídas en un vector plano y procesan la información para clasificar según las categorías establecidas.  
- **`Softmax`**: Calcula la probabilidad de pertenencia a cada clase (Turismo, Pick-Up, Camioneta), permitiendo elegir la categoría con mayor probabilidad.


---

## Tecnologías utilizadas

| Herramienta        | Uso principal                                                                 |
|--------------------|-------------------------------------------------------------------------------|
| 🐍 **Python**       | Lenguaje de programación base utilizado para el desarrollo del proyecto.     |
| 🔶 **TensorFlow / Keras** | Framework principal para construir, entrenar y evaluar la red neuronal. |
| 🔢 **NumPy**        | Manejo eficiente de arreglos multidimensionales y operaciones numéricas.     |
| 📊 **Matplotlib**   | Visualización de datos y métricas del modelo (como precisión o pérdida).     |
| 🎨 **ttkbootstrap**   | Librería de temas y estilos para la creación de interfaces gráficas. |

---

## Instrucciones de instalacion 

**Clonar el repositorio**
```bash
git clone https://github.com/everjosue56/Red-Neuronal-Clasificacion-De-Autos-.git
```
**Instalar librerias**
```bash
pip install tensorflow tensorflow-datasets numpy matplotlib scikit-learn pandas ttkbootstrap
```
**Descargar dataset**
```bash
https://www.kaggle.com/datasets/ademboukhris/cars-body-type-cropped
```

## 🛠️ Estructura del Proyecto

```
cars_classificator/
│
├── dataset/                 # Datasets y datos procesados
│   ├── test/                # Imagenes de prueba
│   ├── train/               # Imagenes de entrenamiento
│   └── valid/               # Imagenes de validación
│
├── models/                  # Modelos guardados (el modelo se crea al ejecutar el entrenamiento)
│
├── results/                 # Graficas y resultados del entrenamiento
│
├── samples/                 # Imagenes nuevas para predicciones
│
├── scripts/                 # Módulos de utilidades
│   ├── load_data.py         # Carga y preprocesamiento de datos
│   ├── model_builder.py     # Script de carga del modelo entrenado
|   ├── model_trainer.py     # Script de entrenamiento
│   └── prediction.py        # Script de predicción
│
├── .gitignore               # Archivo de ignorados
├── main.py                  # Script principal de interfaz gráfica
└── README.md                # Este archivo
```

---

## Instrucciones de ejecución

**1. Cargar y preprocesar el dataset**

```bash
python scripts/load_data.py
```

**2. Crear y entrenar el modelo**

```bash
python scripts/model_trainer.py
```

**3. Cargar el modelo entrenado**

```bash
python scripts/model_builder.py
```

**4. Poner a prueba el modelo**

```bash
python scripts/prediction.py
```
