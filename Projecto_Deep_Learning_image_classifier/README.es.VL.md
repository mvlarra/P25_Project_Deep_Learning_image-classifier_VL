---

## ✅ Proyecto Personal / Implementación Final

> *A continuación se presenta el resumen de la implementación propia desarrollada por Valentina Larrañaga.*

### 📁 Estructura del proyecto

Este proyecto está organizado en **dos notebooks**:

1. `01_data_preparation.ipynb`: descarga y preprocesamiento del dataset.
2. `02_model_training.ipynb`: definición del modelo, entrenamiento y evaluación.

### ⚠️ Consideraciones técnicas

- Debido a limitaciones de hardware (RAM reducida), el tamaño del dataset se redujo temporalmente a 10 imágenes para entrenamiento y validación.
- Esto afecta la estabilidad de las métricas y los gráficos, pero permite validar el flujo completo del modelo en entornos limitados.

### 🧠 Modelo utilizado

Se empleó una CNN inspirada en VGG, con capas `Conv2D`, `MaxPool2D` y `Dense`, finalizando con una neurona de salida y activación `sigmoid` para clasificación binaria.

El entrenamiento se complementó con:

- `ModelCheckpoint`: guarda el mejor modelo durante el entrenamiento.
- `EarlyStopping`: detiene el entrenamiento si no hay mejora.

### 📉 Resultados

El entrenamiento mostró comportamiento inestable en las métricas debido al tamaño reducido de los datos, pero confirmó el correcto funcionamiento del flujo de carga, entrenamiento, evaluación y predicción.

### 🧪 Predicción

Se realizó una prueba de predicción sobre una imagen individual, con el modelo previamente guardado (`mejor_modelo.keras`), obteniendo la probabilidad de que fuera un gato o un perro.

---

## 📬 Contacto

Proyecto desarrollado por Valentina Larrañaga.  
📧 mvlarra@outlook.com  
🌐 [LinkedIn](https://www.linkedin.com/)