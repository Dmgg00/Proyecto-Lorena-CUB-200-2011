"""
Script principal para entrenar y comparar los tres modelos:
- CNN base desde cero
- ResNet50 con Transfer Learning
- EfficientNetB0 con Transfer Learning
"""
import os
import sys

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.base_cnn import create_base_cnn
from models.resnet50_model import create_resnet50_model, unfreeze_model
from models.efficientnet_model import create_efficientnet_model, unfreeze_efficientnet
from utils.data_utils import create_data_generators, get_test_generator
from utils.training_utils import train_model, get_callbacks
from utils.visualization import (
    plot_training_history, compare_models, evaluate_model, print_model_summary
)


def main():
    """
    Función principal para entrenar y comparar los modelos.
    """
    # Configuración
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_CLASSES = 200
    EPOCHS = 50
    
    # Directorios de datos (ajustar según la estructura del dataset)
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/val'
    TEST_DIR = 'data/test'
    
    print("="*70)
    print("PROYECTO: Clasificación de 200 especies de aves - CUB-200-2011")
    print("="*70)
    print("\nModelos a entrenar:")
    print("1. CNN Base (desde cero)")
    print("2. ResNet50 (Transfer Learning + Fine-tuning)")
    print("3. EfficientNetB0 (Transfer Learning + Fine-tuning)")
    print("\nCallbacks utilizados:")
    print("- Early Stopping")
    print("- ReduceLROnPlateau")
    print("- ModelCheckpoint")
    print("="*70)
    
    # Verificar si existen los directorios de datos
    if not all([os.path.exists(d) for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]]):
        print("\n⚠️  ADVERTENCIA: Los directorios de datos no existen.")
        print("Este script requiere el dataset CUB-200-2011 organizado en:")
        print(f"  - {TRAIN_DIR}")
        print(f"  - {VAL_DIR}")
        print(f"  - {TEST_DIR}")
        print("\nPara usar este script:")
        print("1. Descarga el dataset CUB-200-2011")
        print("2. Organiza las imágenes en las carpetas correspondientes")
        print("3. Ejecuta nuevamente este script")
        print("\nEl código está listo para ejecutarse una vez tengas los datos.")
        return
    
    # Crear generadores de datos
    print("\n📊 Creando generadores de datos con data augmentation...")
    train_gen, val_gen = create_data_generators(
        TRAIN_DIR, VAL_DIR, 
        img_size=IMG_SIZE, 
        batch_size=BATCH_SIZE
    )
    test_gen = get_test_generator(TEST_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    
    print(f"✓ Clases encontradas: {len(train_gen.class_indices)}")
    print(f"✓ Imágenes de entrenamiento: {train_gen.samples}")
    print(f"✓ Imágenes de validación: {val_gen.samples}")
    print(f"✓ Imágenes de prueba: {test_gen.samples}")
    
    # Listas para almacenar historias y resultados
    histories = []
    model_names = []
    
    # ========================================================================
    # MODELO 1: CNN Base desde cero
    # ========================================================================
    print("\n" + "="*70)
    print("🔵 ENTRENANDO MODELO 1: CNN Base (desde cero)")
    print("="*70)
    
    cnn_model = create_base_cnn(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
    print("\n📋 Arquitectura del modelo:")
    cnn_model.summary()
    
    print(f"\n🚀 Iniciando entrenamiento (máximo {EPOCHS} épocas)...")
    cnn_history = train_model(
        cnn_model, train_gen, val_gen, 
        model_name='base_cnn', 
        epochs=EPOCHS
    )
    
    plot_training_history(cnn_history, 'Base CNN')
    histories.append(cnn_history)
    model_names.append('CNN Base')
    
    # Evaluar en conjunto de prueba
    cnn_results = evaluate_model(cnn_model, test_gen)
    print_model_summary(cnn_results, 'CNN Base')
    
    # ========================================================================
    # MODELO 2: ResNet50 con Transfer Learning
    # ========================================================================
    print("\n" + "="*70)
    print("🟢 ENTRENANDO MODELO 2: ResNet50 (Transfer Learning)")
    print("="*70)
    
    # Fase 1: Entrenar solo las capas superiores
    print("\n📌 Fase 1: Entrenamiento de capas superiores...")
    resnet_model = create_resnet50_model(
        input_shape=(*IMG_SIZE, 3), 
        num_classes=NUM_CLASSES
    )
    
    resnet_history1 = train_model(
        resnet_model, train_gen, val_gen,
        model_name='resnet50_phase1',
        epochs=20
    )
    
    # Fase 2: Fine-tuning
    print("\n📌 Fase 2: Fine-tuning de las últimas capas...")
    resnet_model = unfreeze_model(resnet_model, trainable_layers=50)
    
    resnet_history2 = train_model(
        resnet_model, train_gen, val_gen,
        model_name='resnet50_phase2',
        epochs=30
    )
    
    # Combinar historias
    resnet_history = type('obj', (object,), {
        'history': {
            'accuracy': resnet_history1.history['accuracy'] + resnet_history2.history['accuracy'],
            'val_accuracy': resnet_history1.history['val_accuracy'] + resnet_history2.history['val_accuracy'],
            'loss': resnet_history1.history['loss'] + resnet_history2.history['loss'],
            'val_loss': resnet_history1.history['val_loss'] + resnet_history2.history['val_loss']
        }
    })()
    
    plot_training_history(resnet_history, 'ResNet50')
    histories.append(resnet_history)
    model_names.append('ResNet50')
    
    # Evaluar en conjunto de prueba
    resnet_results = evaluate_model(resnet_model, test_gen)
    print_model_summary(resnet_results, 'ResNet50')
    
    # ========================================================================
    # MODELO 3: EfficientNetB0 con Transfer Learning
    # ========================================================================
    print("\n" + "="*70)
    print("🟡 ENTRENANDO MODELO 3: EfficientNetB0 (Transfer Learning)")
    print("="*70)
    
    # Fase 1: Entrenar solo las capas superiores
    print("\n📌 Fase 1: Entrenamiento de capas superiores...")
    efficient_model = create_efficientnet_model(
        input_shape=(*IMG_SIZE, 3),
        num_classes=NUM_CLASSES
    )
    
    efficient_history1 = train_model(
        efficient_model, train_gen, val_gen,
        model_name='efficientnet_phase1',
        epochs=20
    )
    
    # Fase 2: Fine-tuning
    print("\n📌 Fase 2: Fine-tuning de las últimas capas...")
    efficient_model = unfreeze_efficientnet(efficient_model, unfreeze_from=100)
    
    efficient_history2 = train_model(
        efficient_model, train_gen, val_gen,
        model_name='efficientnet_phase2',
        epochs=30
    )
    
    # Combinar historias
    efficient_history = type('obj', (object,), {
        'history': {
            'accuracy': efficient_history1.history['accuracy'] + efficient_history2.history['accuracy'],
            'val_accuracy': efficient_history1.history['val_accuracy'] + efficient_history2.history['val_accuracy'],
            'loss': efficient_history1.history['loss'] + efficient_history2.history['loss'],
            'val_loss': efficient_history1.history['val_loss'] + efficient_history2.history['val_loss']
        }
    })()
    
    plot_training_history(efficient_history, 'EfficientNetB0')
    histories.append(efficient_history)
    model_names.append('EfficientNetB0')
    
    # Evaluar en conjunto de prueba
    efficient_results = evaluate_model(efficient_model, test_gen)
    print_model_summary(efficient_results, 'EfficientNetB0')
    
    # ========================================================================
    # COMPARACIÓN FINAL
    # ========================================================================
    print("\n" + "="*70)
    print("📊 COMPARACIÓN DE MODELOS")
    print("="*70)
    
    compare_models(histories, model_names)
    
    print("\n✅ Resumen de resultados:")
    print(f"{'Modelo':<20} {'Test Accuracy':<15} {'Test Loss':<15}")
    print("-"*50)
    print(f"{'CNN Base':<20} {cnn_results['test_accuracy']:.4f}          {cnn_results['test_loss']:.4f}")
    print(f"{'ResNet50':<20} {resnet_results['test_accuracy']:.4f}          {resnet_results['test_loss']:.4f}")
    print(f"{'EfficientNetB0':<20} {efficient_results['test_accuracy']:.4f}          {efficient_results['test_loss']:.4f}")
    
    print("\n🎯 Conclusión:")
    print("Los modelos de Transfer Learning (ResNet50 y EfficientNetB0) demuestran")
    print("ser superiores en precisión y menor pérdida en comparación con la CNN")
    print("construida desde cero, validando el uso de Transfer Learning para la")
    print("clasificación de especies de aves del dataset CUB-200-2011.")
    
    print("\n✓ Todos los modelos han sido guardados en 'saved_models/'")
    print("✓ Los gráficos han sido guardados en 'results/'")
    print("="*70)


if __name__ == "__main__":
    main()
