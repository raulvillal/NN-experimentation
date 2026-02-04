# Neural Network from Scratch - Fashion MNIST

Una implementación completa de una red neuronal desde cero en Python para clasificación de imágenes Fashion MNIST. Este proyecto incluye entrenamiento, guardado de modelos e inferencia interactiva con interfaz gráfica.

## 📚 Referencias

Este proyecto fue desarrollado siguiendo la serie de videos y libro:
- **YouTube:** [Neural Networks from Scratch](https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)

## 🏗️ Estructura del Proyecto

```
NN-experimentation/
├── red_neuronal.py              # Implementación de capas, activaciones, pérdidas y optimizadores
├── training.py                  # Script para entrenar la red neuronal
├── inference.py                 # Interfaz gráfica para inferencia
├── requirements.txt             # Dependencias del proyecto
├── fashion_mnist_images/        # Dataset Fashion MNIST
├── fashion_mnist_model.model    # Modelo entrenado (generado al entrenar)
├── fashion_mnist_model_params.parms  # Parámetros del modelo (generado al entrenar)
└── README.md                    # Este archivo
```

## 🚀 Inicio Rápido

### 1. Configurar el Entorno

```bash
# Crear entorno virtual
python -m venv NN

# Activar entorno (Linux/Mac)
source NN/bin/activate

# Activar entorno (Windows)
NN\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Entrenar el Modelo

```bash
NN/bin/python training.py
```

Este script:
- Carga el dataset Fashion MNIST desde `fashion_mnist_images/`
- Normaliza y prepara los datos
- Crea una red neuronal con 2 capas ocultas (128 neuronas cada una)
- Entrena durante 10 épocas con batch size 128
- Guarda el modelo en `fashion_mnist_model.model`
- Guarda los parámetros en `fashion_mnist_model_params.parms`

### 3. Realizar Inferencia (GUI Interactiva)

```bash
NN/bin/python inference.py
```

Se abrirá una ventana con interfaz gráfica que permite:
- **Load Model**: Cargar un modelo completo (`.model`)
- **Load Params**: Cargar solo los parámetros (`.parms`)
- **Load Image**: Seleccionar una imagen para clasificar
- **Predict**: Ejecutar la predicción

**Procesamiento de imagen:**
- Se convierte a escala de grises (IMREAD_GRAYSCALE)
- Se invierten los colores (`255 - imagen`)
- Se redimensiona a 28×28 píxeles
- Se normaliza a rango [-1, 1]

**Resultados:**
- Se muestra la etiqueta predicha
- Se muestran las 5 clases con mayor confianza

## 📊 Arquitectura de la Red Neuronal

```
Entrada (784) 
    ↓
Capa Densa (784 → 128) + ReLU
    ↓
Capa Densa (128 → 128) + ReLU
    ↓
Capa Densa (128 → 10) + Softmax
    ↓
Salida (10 clases)
```

**Configuración de entrenamiento:**
- **Optimizador:** Adam (decay=1e-5)
- **Pérdida:** Categorical Crossentropy
- **Métrica:** Accuracy Categorical
- **Épocas:** 10
- **Batch Size:** 128

## 📦 Dependencias

Las versiones exactas están especificadas en `requirements.txt`:
- numpy==2.4.0
- opencv-python==4.13.0.90
- nnfs==0.5.1
- matplotlib==3.10.8
- pillow==12.1.0

## 📝 Clases Implementadas

### Capas
- `Layer_Dense`: Capa completamente conectada con regularización L1/L2
- `Layer_Dropout`: Dropout para regularización
- `Layer_Input`: Capa de entrada

### Activaciones
- `Activation_ReLU`: Rectified Linear Unit
- `Activation_Softmax`: Softmax para clasificación multiclase
- `Activation_Sigmoid`: Sigmoid para clasificación binaria
- `Activation_Linear`: Activación lineal para regresión

### Pérdidas
- `Loss_CategorialCrossentropy`: Para clasificación multiclase
- `Loss_BinaryCrossentropy`: Para clasificación binaria
- `Loss_MeanSquaredError`: Para regresión
- `Loss_MeanAbsoluteError`: Para regresión

### Optimizadores
- `Optimizer_SGD`: Descenso de gradiente estocástico con momentum
- `Optimizer_Adagrad`: Adagrad
- `Optimizer_RMSprop`: RMSprop
- `Optimizer_Adam`: Adam

### Métricas
- `Accuracy_Categorical`: Precisión para clasificación multiclase
- `Accuracy_Regression`: Precisión para regresión

## 🎯 Clases Fashion MNIST

El dataset contiene 10 clases de prendas:

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## 💾 Guardando y Cargando Modelos

### Guardar modelo completo (con arquitectura)

```python
model.save('fashion_mnist_model.model')
```

### Guardar solo parámetros (pesos y sesgos)

```python
model.save_parameters('fashion_mnist_model_params.parms')
```

### Cargar modelo completo

```python
model = rn.Model.load('fashion_mnist_model.model')
```

### Cargar desde parámetros

```python
model = build_model_from_params('fashion_mnist_model_params.parms')
```

## 🔧 Personalización

Para modificar la arquitectura, edita `training.py`:

```python
model.add(rn.Layer_Dense(X.shape[1], 256))  # Aumenta neuronas
model.add(rn.Activation_ReLU())
model.add(rn.Layer_Dropout(0.2))  # Añade dropout
model.add(rn.Layer_Dense(256, 128))
model.add(rn.Activation_ReLU())
model.add(rn.Layer_Dense(128, 10))
model.add(rn.Activation_Softmax())
```

## 📄 Licencia

Este proyecto fue desarrollado como material educativo basado en el tutorial de NNFS.

## ✨ Notas

- El script `inference.py` requiere Tkinter para la interfaz gráfica
- Las imágenes deben estar en formato PNG, JPG, JPEG o BMP
- Las imágenes se redimensionan automáticamente a 28×28 píxeles
- El modelo espera imágenes en escala de grises
