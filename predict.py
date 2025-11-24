"""
Script de ejemplo para realizar predicciones con un modelo entrenado.
"""
import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def load_and_predict(model_path, image_path, class_names=None):
    """
    Carga un modelo entrenado y realiza una predicción sobre una imagen.
    
    Args:
        model_path: Ruta al modelo guardado (.h5)
        image_path: Ruta a la imagen para predecir
        class_names: Lista opcional de nombres de clases
        
    Returns:
        predicted_class: Índice de la clase predicha
        confidence: Confianza de la predicción
    """
    # Cargar el modelo
    print(f"Cargando modelo desde: {model_path}")
    model = load_model(model_path)
    
    # Cargar y preprocesar la imagen
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Realizar predicción
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Mostrar resultado
    print(f"\nPredicción:")
    print(f"  - Clase predicha: {predicted_class}")
    if class_names and predicted_class < len(class_names):
        print(f"  - Nombre de la clase: {class_names[predicted_class]}")
    print(f"  - Confianza: {confidence:.2%}")
    
    # Mostrar imagen con predicción
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    title = f"Predicción: Clase {predicted_class}"
    if class_names and predicted_class < len(class_names):
        title = f"Predicción: {class_names[predicted_class]}"
    title += f"\nConfianza: {confidence:.2%}"
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    print("\n✓ Resultado guardado en 'prediction_result.png'")
    plt.close()
    
    return predicted_class, confidence


def predict_top_k(model_path, image_path, k=5, class_names=None):
    """
    Muestra las top-k predicciones para una imagen.
    
    Args:
        model_path: Ruta al modelo guardado (.h5)
        image_path: Ruta a la imagen para predecir
        k: Número de predicciones principales a mostrar
        class_names: Lista opcional de nombres de clases
    """
    # Cargar el modelo
    model = load_model(model_path)
    
    # Cargar y preprocesar la imagen
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Realizar predicción
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Obtener top-k predicciones
    top_k_indices = np.argsort(predictions)[-k:][::-1]
    
    print(f"\nTop-{k} Predicciones:")
    print("-" * 60)
    for i, idx in enumerate(top_k_indices, 1):
        class_name = class_names[idx] if class_names and idx < len(class_names) else f"Clase {idx}"
        print(f"{i}. {class_name}: {predictions[idx]:.2%}")
    print("-" * 60)


def main():
    """
    Función principal para demostrar el uso del script.
    """
    print("="*70)
    print("Script de Predicción - Clasificación de Aves CUB-200-2011")
    print("="*70)
    
    # Ejemplo de uso
    model_path = "saved_models/resnet50_phase2_best.h5"
    image_path = "data/test/example_bird.jpg"
    
    # Verificar que existan los archivos
    if not os.path.exists(model_path):
        print(f"\n⚠️  Modelo no encontrado: {model_path}")
        print("\nPara usar este script:")
        print("1. Entrena un modelo usando train.py")
        print("2. Asegúrate de que el modelo esté guardado en 'saved_models/'")
        print("3. Proporciona la ruta a una imagen de ave")
        print("\nEjemplo de uso programático:")
        print("  from predict import load_and_predict")
        print("  predicted_class, confidence = load_and_predict(")
        print("      'saved_models/resnet50_phase2_best.h5',")
        print("      'ruta/a/imagen.jpg'")
        print("  )")
        return
    
    if not os.path.exists(image_path):
        print(f"\n⚠️  Imagen no encontrada: {image_path}")
        print("Proporciona la ruta a una imagen de ave válida.")
        return
    
    # Realizar predicción
    predicted_class, confidence = load_and_predict(model_path, image_path)
    
    # Mostrar top-5 predicciones
    predict_top_k(model_path, image_path, k=5)
    
    print("\n✓ Predicción completada")
    print("="*70)


if __name__ == "__main__":
    main()
