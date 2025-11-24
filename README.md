# Proyecto de Clasificación de Aves - CUB-200-2011

## Descripción del Proyecto

Este proyecto implementa un sistema de clasificación de aves utilizando técnicas de Deep Learning sobre el dataset CUB-200-2011 (Caltech-UCSD Birds-200-2011). El objetivo principal es desarrollar y comparar diferentes arquitecturas de redes neuronales para identificar automáticamente especies de aves a partir de imágenes.

## Objetivos

- Implementar y evaluar modelos de clasificación de imágenes para 200 especies de aves
- Comparar el rendimiento entre diferentes arquitecturas:
  - CNN desde cero
  - Transfer Learning con ResNet50
  - Transfer Learning con EfficientNetB0
- Aplicar técnicas de regularización para prevenir overfitting
- Optimizar los modelos mediante técnicas como fine-tuning y callbacks

## Dataset: CUB-200-2011

El dataset CUB-200-2011 contiene:
- **11,788 imágenes** de 200 especies diferentes de aves
- Imágenes de alta calidad con anotaciones detalladas
- División en conjuntos de entrenamiento y validación
- Desafíos de clasificación debido a la similitud visual entre especies

## Metodología

### 1. Modelo Base con CNN desde Cero
Implementación de una red neuronal convolucional construida desde cero para establecer una línea base de rendimiento.

### 2. Transfer Learning con ResNet50
Utilización de ResNet50 pre-entrenada en ImageNet, con:
- Congelamiento inicial de capas base
- Fine-tuning de capas superiores
- Callbacks para prevenir overfitting (EarlyStopping, ReduceLROnPlateau)

### 3. Transfer Learning con EfficientNetB0
Implementación de EfficientNetB0, una arquitectura más eficiente que balancea precisión y recursos computacionales.

## Tecnologías Utilizadas

- **Python 3**
- **TensorFlow/Keras** - Framework principal de Deep Learning
- **Google Colab** - Entorno de desarrollo con GPU
- **NumPy, Pandas** - Procesamiento de datos
- **Matplotlib** - Visualización de resultados

## Cómo Usar

### Requisitos Previos

```bash
tensorflow>=2.x
numpy
pandas
matplotlib
```

### Ejecución

1. Abre el notebook en Google Colab haciendo clic en el badge:
   
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dmgg00/Proyecto-Lorena-CUB-200-2011/blob/main/proyecto_lorena.ipynb)

2. Configura el entorno con GPU para mejor rendimiento:
   - Runtime → Change runtime type → GPU

3. Ejecuta las celdas secuencialmente para:
   - Cargar y preparar el dataset
   - Entrenar los modelos
   - Evaluar el rendimiento
   - Visualizar resultados

## Resultados

Los modelos fueron entrenados y evaluados, obteniendo métricas de:
- **Accuracy** (Precisión)
- **Loss** (Pérdida)
- Comparativas entre diferentes arquitecturas
- Análisis de overfitting y convergencia

Este proyecto es de código abierto y está disponible para fines educativos.

## 🙏 Agradecimientos

- Dataset CUB-200-2011 proporcionado por Caltech-UCSD
- Modelos pre-entrenados de TensorFlow/Keras
- Comunidad de Google Colab
