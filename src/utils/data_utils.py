"""
Utilidades para cargar y preprocesar el dataset CUB-200-2011.
"""
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def create_data_generators(train_dir, val_dir, img_size=(224, 224), batch_size=32):
    """
    Crea generadores de datos para entrenamiento y validación con data augmentation.
    
    Args:
        train_dir: Directorio con imágenes de entrenamiento
        val_dir: Directorio con imágenes de validación
        img_size: Tamaño de las imágenes (ancho, alto)
        batch_size: Tamaño del batch
        
    Returns:
        train_generator, val_generator: Generadores de datos
    """
    # Data augmentation para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    # Solo rescaling para validación
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator


def get_test_generator(test_dir, img_size=(224, 224), batch_size=32):
    """
    Crea un generador de datos para el conjunto de prueba.
    
    Args:
        test_dir: Directorio con imágenes de prueba
        img_size: Tamaño de las imágenes (ancho, alto)
        batch_size: Tamaño del batch
        
    Returns:
        test_generator: Generador de datos de prueba
    """
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator
