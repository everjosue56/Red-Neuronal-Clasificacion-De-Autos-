import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import subprocess
import os
import sys
import threading

# Configuración del modelo
IMG_SIZE = (128, 128)
CLASSES = ['Camioneta', 'Pick-Up', 'Turismo'] 

# Variable global para el modelo
model = None
ruta_imagen = None

def cargar_imagen():
    global ruta_imagen
    ruta_imagen = filedialog.askopenfilename(
        filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg")]
    )
    if ruta_imagen:
        img = Image.open(ruta_imagen)
        img = img.resize((250, 250))  # Tamaño para visualización
        img_tk = ImageTk.PhotoImage(img)
        label_imagen.config(image=img_tk)
        label_imagen.image = img_tk
        label_resultado.config(text="")

def preprocesar_imagen(ruta):
    """Preprocesa la imagen para que sea compatible con el modelo"""
    img = tf.keras.preprocessing.image.load_img(ruta, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión batch
    img_array = img_array / 255.0  # Normalizar como en el entrenamiento
    return img_array

def verificar_imagen():
    if not ruta_imagen:
        label_resultado.config(text="Primero carga una imagen", bootstyle="danger")
        return
    
    if model is None:
        label_resultado.config(text="Error: Modelo no cargado", bootstyle="danger")
        return
    
    try:
        # Preprocesar la imagen
        img_array = preprocesar_imagen(ruta_imagen)
        
        # Hacer la predicción
        pred = model.predict(img_array)
        predicted_class = CLASSES[np.argmax(pred)]
        confidence = np.max(pred) * 100  # Porcentaje de confianza
        
        # Mostrar resultado
        resultado = f"Categoría: {predicted_class}\nConfianza: {confidence:.2f}%"
        label_resultado.config(text=resultado, bootstyle="success")
        
    except Exception as e:
        label_resultado.config(text=f"Error: {str(e)}", bootstyle="danger")

def ejecutar_script(script_path, progress_bar):
    """Ejecuta un script Python externo con barra de progreso"""
    def ejecutar():
        try:
            # Verificar si el archivo existe
            if not os.path.exists(script_path):
                messagebox.showerror("Error", f"El archivo {script_path} no existe")
                return
            
            # Configurar barra de progreso indeterminada
            progress_bar.start()
            ventana.update()
            
            # Ejecutar el script
            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
            
            # Detener barra de progreso
            progress_bar.stop()
            progress_bar['value'] = 0
            
            # Mostrar resultados
            if result.returncode == 0:
                messagebox.showinfo("Éxito", f"Script ejecutado correctamente:\n{result.stdout}")
                
                # Si es el script de carga de modelo, actualizamos la variable global
                if "model_builder.py" in script_path:
                    cargar_modelo()
            else:
                messagebox.showerror("Error", f"Error al ejecutar el script:\n{result.stderr}")
                
        except Exception as e:
            progress_bar.stop()
            progress_bar['value'] = 0
            messagebox.showerror("Error", f"Excepción al ejecutar el script:\n{str(e)}")
    
    # Ejecutar en un hilo separado para no bloquear la interfaz
    threading.Thread(target=ejecutar, daemon=True).start()

def cargar_dataset():
    """Ejecuta el script de carga de dataset"""
    ejecutar_script("scripts/load_data.py", progress_bar_dataset)

def entrenar_modelo():
    """Ejecuta el script de entrenamiento del modelo"""
    # Preguntar confirmación ya que el entrenamiento puede tomar tiempo
    if messagebox.askyesno("Confirmar", "¿Estás seguro de querer entrenar el modelo? Esto puede tomar varios minutos."):
        ejecutar_script("scripts/model_trainer.py", progress_bar_entrenamiento)

def cargar_modelo():
    """Carga el modelo en memoria"""
    global model
    try:
        progress_bar_modelo.start()
        ventana.update()
        
        model = tf.keras.models.load_model("models/cars_classificator.h5")
        messagebox.showinfo("Éxito", "Modelo cargado correctamente en memoria")
        label_estado_modelo.config(text="Estado: Modelo cargado ✅", bootstyle="success")
    except Exception as e:
        model = None
        messagebox.showerror("Error", f"No se pudo cargar el modelo:\nNo se encontró el archivo 'models/cars_classificator.h5'")
        label_estado_modelo.config(text="Estado: Modelo NO cargado ❌", bootstyle="danger")
    finally:
        progress_bar_modelo.stop()
        progress_bar_modelo['value'] = 0

# Configuración de la ventana
ventana = ttk.Window(themename="superhero")
ventana.title("Red Neuronal - Clasificador de Vehículos")
ventana.geometry("720x950")  # Aumentamos el tamaño para las barras de progreso

# Titulo
titulo = ttk.Label(ventana, text="Clasificador de Vehiculos 🚗", font=("Segoe UI", 18, "bold"))
titulo.pack(pady=15)

# Sección de clasificación de imágenes
frame_clasificacion = ttk.Frame(ventana)
frame_clasificacion.pack(pady=10, fill="x", padx=20)

btn_cargar = ttk.Button(
    frame_clasificacion, 
    text="📂 Cargar Imagen", 
    command=cargar_imagen, 
    bootstyle="primary",
    width=25 
)
btn_cargar.pack(pady=10, ipady=8)

label_imagen = ttk.Label(frame_clasificacion)
label_imagen.pack()

btn_verificar = ttk.Button(
    frame_clasificacion, 
    text="🔍 Verificar", 
    command=verificar_imagen, 
    bootstyle="info",
    width=25
)
btn_verificar.pack(pady=10, ipady=8)

label_resultado = ttk.Label(frame_clasificacion, text="", font=("Segoe UI", 14, "bold"))
label_resultado.pack(pady=10)

# Separador
separador = ttk.Separator(ventana, bootstyle="white")
separador.pack(fill="x", pady=10, padx=20)

# Sección de gestión del modelo
frame_modelo = ttk.Frame(ventana)
frame_modelo.pack(pady=10, fill="x", padx=20)

ttk.Label(frame_modelo, text="Gestión del Modelo", font=("Segoe UI", 14)).pack(pady=5)

label_estado_modelo = ttk.Label(
    frame_modelo, 
    text="Estado: Modelo NO cargado ❌", 
    font=("Segoe UI", 10),
    bootstyle="danger"
)
label_estado_modelo.pack(pady=5)

# Botones de gestión
btn_frame = ttk.Frame(frame_modelo)
btn_frame.pack(pady=10)

btn_cargar_dataset = ttk.Button(
    btn_frame, 
    text="📊 Cargar Dataset", 
    command=cargar_dataset, 
    bootstyle="light",
    width=20
)
btn_cargar_dataset.grid(row=0, column=0, padx=5, pady=5, ipady=5)

btn_entrenar = ttk.Button(
    btn_frame, 
    text="⚙️ Entrenar Modelo", 
    command=entrenar_modelo, 
    bootstyle="warning",
    width=20
)
btn_entrenar.grid(row=0, column=1, padx=5, pady=5, ipady=5)

btn_cargar_modelo = ttk.Button(
    btn_frame, 
    text="🤖 Cargar Modelo", 
    command=cargar_modelo, 
    bootstyle="success",
    width=20
)
btn_cargar_modelo.grid(row=0, column=2, padx=5, pady=5, ipady=5)

# Barras de progreso para cada operación
progress_frame = ttk.Frame(frame_modelo)
progress_frame.pack(pady=10)

# Barra para cargar dataset
ttk.Label(progress_frame, text="Progreso Carga Dataset:").grid(row=0, column=0, sticky="w", padx=5)
progress_bar_dataset = ttk.Progressbar(
    progress_frame,
    orient="horizontal",
    mode="indeterminate",
    length=400,
    bootstyle="light-striped"
)
progress_bar_dataset.grid(row=1, column=0, pady=5, padx=5)

# Barra para entrenamiento
ttk.Label(progress_frame, text="Progreso Entrenamiento:").grid(row=2, column=0, sticky="w", padx=5)
progress_bar_entrenamiento = ttk.Progressbar(
    progress_frame,
    orient="horizontal",
    mode="indeterminate",
    length=400,
    bootstyle="warning-striped"
)
progress_bar_entrenamiento.grid(row=3, column=0, pady=5, padx=5)

# Barra para cargar modelo
ttk.Label(progress_frame, text="Progreso Carga Modelo:").grid(row=4, column=0, sticky="w", padx=5)
progress_bar_modelo = ttk.Progressbar(
    progress_frame,
    orient="horizontal",
    mode="indeterminate",
    length=400,
    bootstyle="success-striped"
)
progress_bar_modelo.grid(row=5, column=0, pady=5, padx=5)

ventana.mainloop()