<!-- hide -->
# RNA para clasificación de imágenes - Guía paso a paso
<!-- endhide -->

- Comprender un dataset nuevo.
- Modelar los datos utilizando una RNA.
- Analizar los resultados y optimizar el modelo.

## 🌱 Cómo iniciar este proyecto

Sigue las siguientes instrucciones:

1. Crea un nuevo repositorio basado en el [proyecto de Machine Learning](https://github.com/4GeeksAcademy/machine-learning-python-template) o [haciendo clic aquí](https://github.com/4GeeksAcademy/machine-learning-python-template/generate).
2. Abre el repositorio creado recientemente en Codespace usando la [extensión del botón de Codespace](https://docs.github.com/en/codespaces/developing-in-codespaces/creating-a-codespace-for-a-repository#creating-a-codespace-for-a-repository).
3. Una vez que el VSCode del Codespace haya terminado de abrirse, comienza tu proyecto siguiendo las instrucciones a continuación.

## 🚛 Cómo entregar este proyecto

Una vez que hayas terminado de resolver el caso práctico, asegúrate de confirmar tus cambios, haz push a tu repositorio y ve a 4Geeks.com para subir el enlace del repositorio.

## 📝 Instrucciones

### Sistema de clasificación de imágenes

El conjunto de datos se compone de fotos de perros y gatos proporcionadas como un subconjunto de fotos de uno mucho más grande de 3 millones de fotos anotadas manualmente. Estos datos se obtuvieron a través de una colaboración entre Petfinder.com y Microsoft.

El conjunto de datos se usó originalmente como un CAPTCHA, es decir, una tarea que se cree que un humano encuentra trivial, pero que una máquina no puede resolver, que se usa en sitios web para distinguir entre usuarios humanos y bots. La tarea se denominó "Asirra". Cuando se presentó "Asirra", se mencionó "que los estudios de usuarios indican que los humanos pueden resolverlo el 99,6% de las veces en menos de 30 segundos". A menos que se produzca un gran avance en la visión artificial, esperamos que los ordenadores no tengan más de 1/54.000 posibilidades de resolverlo.

En el momento en que se publicó la competencia, el resultado de última generación se logró con un SVM y se describió en un artículo de 2007 con el título "Ataques de Machine Learning contra el CAPTCHA de Asirra" (PDF) que logró una precisión de clasificación del 80%. Fue este documento el que demostró que la tarea ya no era una tarea adecuada para un CAPTCHA poco después de que se propusiera la tarea.

#### Paso 1: Carga del conjunto de datos

El conjunto de datos se encuentra en Kaggle y tendrás que acceder a ella para descargarlos. La competición la puedes encontrar [aquí](https://www.kaggle.com/c/dogs-vs-cats/data) (o copiando y pegando el siguiente enlace en tu navegador: `https://www.kaggle.com/c/dogs-vs-cats/data`)

Descarga la carpeta dataset y descomprime los archivos. Ahora tendrás una carpeta llamada `train` que contiene 25.000 archivos de imagen (formato .jpg) de perros y gatos. Las fotos están etiquetadas por su nombre de archivo, con la palabra `dog` o `cat`.

#### Paso 2: Visualiza la información de entrada

El primer paso cuando nos enfrentamos a un problema de clasificación de imágenes es obtener toda la información posible a través de ellas. Por lo tanto, carga e imprime las primeras nueve fotos de perros en una sola figura. Repite lo mismo para los gatos. Puedes ver que las fotos son a color y tienen diferentes formas y tamaños.

Esta variedad de tamaños y formatos debe solucionarse antes de entrenar el modelo. Asegúrate de que todas tengan un tamaño fijo de 200x200 píxeles.

Como podrás ver, son una gran cantidad de imágenes, asegúrate de seguir las siguientes normas:

1. **Si tienes más de 12 gigabytes de RAM**, usa la API de procesamiento de imágenes de Keras para cargar las 25.000 fotos en el conjunto de datos de entrenamiento y remodelarlas a fotos cuadradas de 200×200 píxeles. La etiqueta también debe determinarse para cada foto en función de los nombres de archivo. Se debe guardar una tupla de fotos y etiquetas.
2. **Si no tienes más de 12 gigabytes de RAM**, carga las imágenes progresivamente usando la clase Keras `ImageDataGenerator` y la función `flow_from_directory()`. Esto será más lento de ejecutar, pero se ejecutará en hardware de menor capacidad. Esta función prefiere que los datos se dividan en directorios *train* y *test* separados, y debajo de cada directorio para tener un subdirectorio para cada clase.

Una vez tengas todas las imágenes procesadas, crea un objeto `ImageDataGenerator` para datos de entrenamiento y prueba. Luego pasa la carpeta que tiene datos de entrenamiento al objeto `trdata` y, de manera similar, pasa la carpeta que tiene datos de prueba al objeto `tsdata`. De esta forma, se etiquetarán las imágenes automáticamente y estará todo listo para entrar a la red.

#### Paso 3: Construye una RNA

Cualquier clasificador que se ajuste a este problema tendrá que ser robusto porque algunas imágenes muestran al gato o al perro en una esquina o tal vez a 2 gatos o perros en la misma foto. Si has podido investigar algunas de las implementaciones de los ganadores de otras competiciones también relacionadas con imágenes, verás que `VGG16` es una arquitectura de CNN utilizada para ganar la competencia de Kaggle ILSVR (Imagenet) en 2014. Se considera una de las arquitecturas de modelos de visión con mejores resultados hasta la fecha.

Utiliza la siguiente arquitectura de prueba:

```py
model = Sequential()
model.add(Conv2D(input_shape = (224,224,3), filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = "same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2),strides = (2,2)))
model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2),strides = (2,2)))
model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2),strides = (2,2)))
model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2),strides = (2,2)))
model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2),strides = (2,2)))
model.add(Flatten())
model.add(Dense(units = 4096,activation = "relu"))
model.add(Dense(units = 4096,activation = "relu"))
model.add(Dense(units = 2, activation = "softmax"))
```

El código anterior aplica convoluciones a los datos (capas `Conv2D` y `MaxPool2D`) y después aplica capas densas (capas `Dense`) para el procesamiento de los valores numéricos obtenidos tras las convoluciones.

A continuación añade los elementos restantes para conformar el modelo, entrénalo y mide su rendimiento.

#### Paso 4: Optimiza el modelo anterior

Importa el método `ModelCheckpoint` y `EarlyStopping` de Keras. Crea un objeto de ambos y pásalo como funciones callback a `fit_generator`.

Carga el mejor modelo de los anteriores y utiliza el conjunto de test para hacer predicciones.

#### Paso 5: Guarda el modelo

Almacena el modelo en la carpeta correspondiente.

> Nota: También incorporamos muestras de solución en `./solution.ipynb` que te sugerimos honestamente que solo uses si estás atascado por más de 30 minutos o si ya has terminado y quieres compararlo con tu enfoque.

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