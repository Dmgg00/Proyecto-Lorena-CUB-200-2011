# Proyecto de Clasificación de Aves - CUB-200-2011

Este proyecto implementa la clasificación automatizada de 200 especies de aves del dataset CUB-200-2011 mediante técnicas de Deep Learning. Se desarrollaron y compararon tres modelos diferentes para evaluar su rendimiento en la tarea de clasificación.

## 📋 Descripción del Proyecto

El objetivo principal es clasificar automáticamente 200 especies de aves utilizando el dataset Caltech-UCSD Birds-200-2011 (CUB-200-2011). Se implementaron tres arquitecturas de redes neuronales:

1. **CNN Base**: Una red neuronal convolucional construida desde cero
2. **ResNet50**: Modelo pre-entrenado con Transfer Learning y Fine-tuning
3. **EfficientNetB0**: Modelo pre-entrenado con Transfer Learning y Fine-tuning

### Técnicas Implementadas

- **Transfer Learning**: Aprovechamiento de modelos pre-entrenados en ImageNet
- **Fine-tuning**: Ajuste fino de las últimas capas de los modelos pre-entrenados
- **Data Augmentation**: Aumento de datos para mejorar la generalización
- **Early Stopping**: Detención temprana para evitar sobreajuste
- **ReduceLROnPlateau**: Reducción adaptativa de la tasa de aprendizaje

## 🏗️ Estructura del Proyecto

```
Proyecto-Lorena-CUB-200-2011/
├── src/
│   ├── models/
│   │   ├── base_cnn.py           # CNN base desde cero
│   │   ├── resnet50_model.py     # ResNet50 con Transfer Learning
│   │   └── efficientnet_model.py # EfficientNetB0 con Transfer Learning
│   └── utils/
│       ├── data_utils.py         # Carga y preprocesamiento de datos
│       ├── training_utils.py     # Utilidades de entrenamiento y callbacks
│       └── visualization.py      # Visualización de resultados
├── data/
│   ├── train/                    # Imágenes de entrenamiento
│   ├── val/                      # Imágenes de validación
│   └── test/                     # Imágenes de prueba
├── notebooks/                    # Jupyter notebooks para análisis
├── saved_models/                 # Modelos entrenados guardados
├── results/                      # Gráficos y resultados
├── train.py                      # Script principal de entrenamiento
└── requirements.txt              # Dependencias del proyecto
```

## 🚀 Instalación

### Prerequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- GPU con CUDA (recomendado para entrenamiento)

### Pasos de Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/Dmgg00/Proyecto-Lorena-CUB-200-2011.git
cd Proyecto-Lorena-CUB-200-2011
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

3. Descargar el dataset CUB-200-2011:
   - Descargar desde [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/)
   - Extraer y organizar las imágenes en las carpetas `data/train/`, `data/val/` y `data/test/`

## 📊 Dataset

El dataset CUB-200-2011 contiene:
- **200 categorías** de especies de aves
- **11,788 imágenes** en total
- Imágenes con variedad en pose, iluminación y fondo
- Anotaciones detalladas (no utilizadas en este proyecto básico)

### Organización de Datos

Las imágenes deben estar organizadas en la siguiente estructura:
```
data/
├── train/
│   ├── clase_001/
│   │   ├── imagen1.jpg
│   │   └── imagen2.jpg
│   ├── clase_002/
│   └── ...
├── val/
│   └── (misma estructura)
└── test/
    └── (misma estructura)
```

## 🎯 Uso

### Entrenamiento de Modelos

Para entrenar los tres modelos y compararlos:

```bash
python train.py
```

Este script ejecutará:
1. Entrenamiento de la CNN base (desde cero)
2. Entrenamiento de ResNet50 con Transfer Learning y Fine-tuning
3. Entrenamiento de EfficientNetB0 con Transfer Learning y Fine-tuning
4. Comparación de resultados
5. Generación de gráficos y métricas

### Uso de Modelos Individuales

#### CNN Base
```python
from src.models.base_cnn import create_base_cnn

model = create_base_cnn(input_shape=(224, 224, 3), num_classes=200)
```

#### ResNet50
```python
from src.models.resnet50_model import create_resnet50_model, unfreeze_model

# Crear modelo
model = create_resnet50_model(input_shape=(224, 224, 3), num_classes=200)

# Entrenar capas superiores primero, luego aplicar fine-tuning
model = unfreeze_model(model, trainable_layers=50)
```

#### EfficientNetB0
```python
from src.models.efficientnet_model import create_efficientnet_model, unfreeze_efficientnet

# Crear modelo
model = create_efficientnet_model(input_shape=(224, 224, 3), num_classes=200)

# Entrenar capas superiores primero, luego aplicar fine-tuning
model = unfreeze_efficientnet(model, unfreeze_from=100)
```

## 🔍 Modelos Implementados

### 1. CNN Base (Desde Cero)

Arquitectura personalizada con:
- 4 bloques convolucionales con BatchNormalization
- MaxPooling y Dropout para regularización
- Capas densas con 512 y 256 neuronas
- Total: ~10M parámetros entrenables

### 2. ResNet50 (Transfer Learning)

- Base: ResNet50 pre-entrenada en ImageNet
- Capas personalizadas: GlobalAveragePooling + Dense layers
- Fine-tuning de las últimas 50 capas
- Optimizador: Adam con learning rate adaptativo

### 3. EfficientNetB0 (Transfer Learning)

- Base: EfficientNetB0 pre-entrenada en ImageNet
- Capas personalizadas: GlobalAveragePooling + Dense layers
- Fine-tuning de las últimas 100+ capas
- Optimizador: Adam con learning rate adaptativo

## 📈 Resultados Esperados

Los modelos de Transfer Learning (ResNet50 y EfficientNetB0) demostraron ser superiores en:
- **Mayor precisión** en el conjunto de validación y prueba
- **Menor pérdida** durante el entrenamiento
- **Convergencia más rápida** comparado con la CNN base
- **Mejor generalización** gracias al conocimiento pre-entrenado

### Callbacks Utilizados

1. **Early Stopping**
   - Monitor: val_loss
   - Paciencia: 10 épocas
   - Restaura los mejores pesos

2. **ReduceLROnPlateau**
   - Monitor: val_loss
   - Factor de reducción: 0.5
   - Paciencia: 5 épocas

3. **ModelCheckpoint**
   - Guarda el mejor modelo según val_accuracy

## 🛠️ Tecnologías Utilizadas

- **TensorFlow/Keras**: Framework de Deep Learning
- **NumPy**: Computación numérica
- **Pandas**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización
- **Scikit-learn**: Métricas y preprocesamiento
- **Pillow**: Procesamiento de imágenes

## 📝 Notas

- El entrenamiento puede tomar varias horas dependiendo del hardware
- Se recomienda usar GPU para acelerar el entrenamiento
- Los modelos pre-entrenados se descargan automáticamente la primera vez
- Los checkpoints se guardan en `saved_models/`
- Los gráficos se guardan en `results/`

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible para fines educativos.

## 👥 Autor

- Proyecto Lorena - Clasificación de Aves CUB-200-2011

## 🙏 Agradecimientos

- Dataset CUB-200-2011 por Caltech-UCSD
- Comunidad de TensorFlow y Keras
- Investigadores de Transfer Learning y arquitecturas de CNN