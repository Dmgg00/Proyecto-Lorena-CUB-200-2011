"""
Utilidades para visualización de resultados del entrenamiento.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os


def plot_training_history(history, model_name):
    """
    Grafica la historia del entrenamiento (pérdida y precisión).
    
    Args:
        history: Historia del entrenamiento de Keras
        model_name: Nombre del modelo para el título
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfica de precisión
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title(f'{model_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Gráfica de pérdida
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title(f'{model_name} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Guardar figura
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def compare_models(histories, model_names):
    """
    Compara el rendimiento de múltiples modelos.
    
    Args:
        histories: Lista de historias de entrenamiento
        model_names: Lista de nombres de modelos
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Comparar precisión de validación
    for history, name in zip(histories, model_names):
        axes[0].plot(history.history['val_accuracy'], label=name)
    axes[0].set_title('Comparación de Accuracy de Validación')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Comparar pérdida de validación
    for history, name in zip(histories, model_names):
        axes[1].plot(history.history['val_loss'], label=name)
    axes[1].set_title('Comparación de Loss de Validación')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Guardar figura
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model(model, test_generator):
    """
    Evalúa un modelo en el conjunto de prueba.
    
    Args:
        model: Modelo de Keras
        test_generator: Generador de datos de prueba
        
    Returns:
        results: Diccionario con métricas de evaluación
    """
    # Evaluar el modelo
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    
    # Predecir clases
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Obtener clases reales
    true_classes = test_generator.classes
    
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes
    }
    
    return results


def print_model_summary(results, model_name):
    """
    Imprime un resumen de los resultados del modelo.
    
    Args:
        results: Diccionario con métricas de evaluación
        model_name: Nombre del modelo
    """
    print(f"\n{'='*50}")
    print(f"Resultados de {model_name}")
    print(f"{'='*50}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"{'='*50}\n")
