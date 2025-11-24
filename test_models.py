"""
Tests básicos para validar que los modelos se pueden crear correctamente.
"""
import sys
import os

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.base_cnn import create_base_cnn
from models.resnet50_model import create_resnet50_model, unfreeze_model
from models.efficientnet_model import create_efficientnet_model, unfreeze_efficientnet


def test_base_cnn_creation():
    """Test: Crear modelo CNN base."""
    print("Test 1: Creación de CNN Base...")
    try:
        model = create_base_cnn(input_shape=(224, 224, 3), num_classes=200)
        assert model is not None
        assert len(model.layers) > 0
        print("  ✓ CNN Base creada correctamente")
        print(f"    - Parámetros totales: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"  ✗ Error al crear CNN Base: {e}")
        return False


def test_resnet50_creation():
    """Test: Crear modelo ResNet50."""
    print("\nTest 2: Creación de ResNet50...")
    try:
        model = create_resnet50_model(input_shape=(224, 224, 3), num_classes=200)
        assert model is not None
        assert len(model.layers) > 0
        print("  ✓ ResNet50 creada correctamente")
        print(f"    - Parámetros totales: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"  ✗ Error al crear ResNet50: {e}")
        return False


def test_efficientnet_creation():
    """Test: Crear modelo EfficientNetB0."""
    print("\nTest 3: Creación de EfficientNetB0...")
    try:
        model = create_efficientnet_model(input_shape=(224, 224, 3), num_classes=200)
        assert model is not None
        assert len(model.layers) > 0
        print("  ✓ EfficientNetB0 creada correctamente")
        print(f"    - Parámetros totales: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"  ✗ Error al crear EfficientNetB0: {e}")
        return False


def test_model_compilation():
    """Test: Verificar que los modelos están compilados."""
    print("\nTest 4: Verificación de compilación de modelos...")
    try:
        cnn = create_base_cnn(input_shape=(224, 224, 3), num_classes=200)
        assert cnn.optimizer is not None
        print("  ✓ CNN Base compilada correctamente")
        
        resnet = create_resnet50_model(input_shape=(224, 224, 3), num_classes=200)
        assert resnet.optimizer is not None
        print("  ✓ ResNet50 compilada correctamente")
        
        efficient = create_efficientnet_model(input_shape=(224, 224, 3), num_classes=200)
        assert efficient.optimizer is not None
        print("  ✓ EfficientNetB0 compilada correctamente")
        
        return True
    except Exception as e:
        print(f"  ✗ Error en compilación: {e}")
        return False


def test_unfreeze_functions():
    """Test: Verificar funciones de fine-tuning."""
    print("\nTest 5: Verificación de funciones de fine-tuning...")
    try:
        # Test ResNet50 unfreeze
        resnet = create_resnet50_model(input_shape=(224, 224, 3), num_classes=200)
        resnet = unfreeze_model(resnet, trainable_layers=50)
        assert resnet.optimizer is not None
        print("  ✓ ResNet50 fine-tuning configurado correctamente")
        
        # Test EfficientNet unfreeze
        efficient = create_efficientnet_model(input_shape=(224, 224, 3), num_classes=200)
        efficient = unfreeze_efficientnet(efficient, unfreeze_from=100)
        assert efficient.optimizer is not None
        print("  ✓ EfficientNetB0 fine-tuning configurado correctamente")
        
        return True
    except Exception as e:
        print(f"  ✗ Error en fine-tuning: {e}")
        return False


def main():
    """Ejecutar todos los tests."""
    print("="*70)
    print("Ejecutando tests de validación de modelos")
    print("="*70)
    
    results = []
    
    # Ejecutar tests
    results.append(test_base_cnn_creation())
    results.append(test_resnet50_creation())
    results.append(test_efficientnet_creation())
    results.append(test_model_compilation())
    results.append(test_unfreeze_functions())
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE TESTS")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Tests pasados: {passed}/{total}")
    
    if passed == total:
        print("✓ Todos los tests pasaron correctamente")
        print("\nLos modelos están listos para ser entrenados.")
        print("Usa 'python train.py' para comenzar el entrenamiento.")
        return 0
    else:
        print("✗ Algunos tests fallaron")
        print("\nRevisa los errores anteriores para más detalles.")
        return 1


if __name__ == "__main__":
    exit(main())
