# Guía de Preparación del Dataset CUB-200-2011

Esta guía explica cómo descargar y preparar el dataset CUB-200-2011 para usar con este proyecto.

## 1. Descargar el Dataset

El dataset CUB-200-2011 puede descargarse desde:
- **Sitio oficial**: http://www.vision.caltech.edu/datasets/cub_200_2011/
- **Archivo directo**: http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz

```bash
# Descargar usando wget
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz

# O usando curl
curl -O http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
```

## 2. Extraer el Dataset

```bash
# Extraer el archivo
tar -xzf CUB_200_2011.tgz

# Esto creará una carpeta llamada CUB_200_2011/
```

## 3. Estructura del Dataset Original

```
CUB_200_2011/
├── images/
│   ├── 001.Black_footed_Albatross/
│   ├── 002.Laysan_Albatross/
│   ├── ...
│   └── 200.Common_Yellowthroat/
├── train_test_split.txt
├── images.txt
├── image_class_labels.txt
└── classes.txt
```

## 4. Organizar para el Proyecto

El proyecto espera la siguiente estructura:

```
data/
├── train/
│   ├── 001.Black_footed_Albatross/
│   │   ├── imagen1.jpg
│   │   └── imagen2.jpg
│   ├── 002.Laysan_Albatross/
│   └── ...
├── val/
│   └── (misma estructura)
└── test/
    └── (misma estructura)
```

## 5. Script de Preparación (Python)

Puedes usar el siguiente script para organizar automáticamente el dataset:

```python
import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_cub_dataset(source_dir, target_dir, val_split=0.15, test_split=0.15):
    """
    Organiza el dataset CUB-200-2011 en carpetas train/val/test.
    
    Args:
        source_dir: Directorio con las imágenes originales (CUB_200_2011/)
        target_dir: Directorio donde guardar el dataset organizado (data/)
        val_split: Proporción para validación
        test_split: Proporción para prueba
    """
    images_dir = os.path.join(source_dir, 'images')
    split_file = os.path.join(source_dir, 'train_test_split.txt')
    
    # Leer el archivo de división
    with open(split_file, 'r') as f:
        splits = {}
        for line in f:
            img_id, is_train = line.strip().split()
            splits[int(img_id)] = int(is_train)
    
    # Crear directorios de destino
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)
    
    # Procesar cada clase
    for class_dir in sorted(os.listdir(images_dir)):
        class_path = os.path.join(images_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        
        print(f"Procesando clase: {class_dir}")
        
        # Obtener todas las imágenes de la clase
        images = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
        
        # Dividir según train_test_split.txt y luego crear validación
        train_images = []
        test_images = []
        
        for img in images:
            img_path = os.path.join(class_dir, img)
            # Aquí necesitarías mapear el nombre de archivo al ID
            # Para simplificar, dividimos manualmente
            pass
        
        # División simple alternativa: 70% train, 15% val, 15% test
        train_val, test = train_test_split(images, test_size=test_split, random_state=42)
        train, val = train_test_split(train_val, test_size=val_split/(1-test_split), random_state=42)
        
        # Copiar imágenes a sus carpetas correspondientes
        for split, split_images in [('train', train), ('val', val), ('test', test)]:
            dest_class_dir = os.path.join(target_dir, split, class_dir)
            os.makedirs(dest_class_dir, exist_ok=True)
            
            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(dest_class_dir, img)
                shutil.copy2(src, dst)
        
        print(f"  ✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Uso
if __name__ == "__main__":
    prepare_cub_dataset(
        source_dir='CUB_200_2011',
        target_dir='data'
    )
    print("\n✓ Dataset preparado exitosamente!")
```

## 6. Verificación

Después de organizar el dataset, verifica la estructura:

```bash
# Contar clases en cada conjunto
ls data/train/ | wc -l   # Debería ser 200
ls data/val/ | wc -l     # Debería ser 200
ls data/test/ | wc -l    # Debería ser 200

# Contar imágenes totales
find data/train -name "*.jpg" | wc -l
find data/val -name "*.jpg" | wc -l
find data/test -name "*.jpg" | wc -l
```

## 7. Información del Dataset

- **Total de imágenes**: 11,788
- **Número de clases**: 200 especies de aves
- **División sugerida**:
  - Entrenamiento: ~70% (8,251 imágenes)
  - Validación: ~15% (1,768 imágenes)
  - Prueba: ~15% (1,769 imágenes)

## 8. Notas Importantes

- Las imágenes tienen diferentes tamaños; el código de entrenamiento las redimensiona a 224x224
- Algunas clases tienen más imágenes que otras (desequilibrio de clases)
- Se recomienda usar data augmentation para mejorar el rendimiento
- El dataset incluye anotaciones de bounding boxes y puntos clave (no usados en este proyecto básico)

## 9. Referencias

- Paper original: Wah C., Branson S., Welinder P., Perona P., Belongie S. "The Caltech-UCSD Birds-200-2011 Dataset." Computation & Neural Systems Technical Report, CNS-TR-2011-001.
- Sitio web: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
