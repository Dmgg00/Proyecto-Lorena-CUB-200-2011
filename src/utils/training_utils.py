"""
Utilidades para el entrenamiento de modelos con callbacks.
"""
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os


def get_callbacks(model_name, patience=10):
    """
    Crea los callbacks para el entrenamiento: Early Stopping, ReduceLROnPlateau y ModelCheckpoint.
    
    Args:
        model_name: Nombre del modelo para guardar checkpoints
        patience: Paciencia para Early Stopping
        
    Returns:
        callbacks: Lista de callbacks
    """
    # Crear directorio para guardar modelos si no existe
    os.makedirs('saved_models', exist_ok=True)
    
    # Early Stopping - detiene el entrenamiento si no hay mejora
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # ReduceLROnPlateau - reduce la tasa de aprendizaje cuando la métrica deja de mejorar
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # ModelCheckpoint - guarda el mejor modelo
    checkpoint = ModelCheckpoint(
        filepath=f'saved_models/{model_name}_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    return [early_stopping, reduce_lr, checkpoint]


def train_model(model, train_generator, val_generator, model_name, epochs=50):
    """
    Entrena un modelo con los callbacks configurados.
    
    Args:
        model: Modelo de Keras a entrenar
        train_generator: Generador de datos de entrenamiento
        val_generator: Generador de datos de validación
        model_name: Nombre del modelo
        epochs: Número máximo de épocas
        
    Returns:
        history: Historia del entrenamiento
    """
    callbacks = get_callbacks(model_name)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
