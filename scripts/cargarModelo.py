from tensorflow.keras.models import load_model

# Cargar el modelo previamente guardado
modelo_cargado = load_model("modelos/modelo_carros.h5")

# Confirmar carga
print("------- Modelo cargado correctamente. ---------")
