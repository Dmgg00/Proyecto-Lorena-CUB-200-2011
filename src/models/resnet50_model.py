"""
Modelo ResNet50 con Transfer Learning y Fine-tuning para clasificación de aves.
"""
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam


def create_resnet50_model(input_shape=(224, 224, 3), num_classes=200, trainable_layers=50):
    """
    Crea un modelo ResNet50 con Transfer Learning.
    
    Args:
        input_shape: Forma de las imágenes de entrada (alto, ancho, canales)
        num_classes: Número de clases (especies de aves)
        trainable_layers: Número de capas a entrenar desde el final (-1 para todas)
        
    Returns:
        model: Modelo ResNet50 compilado
    """
    # Cargar ResNet50 pre-entrenado sin la capa superior
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Congelar las capas base inicialmente
    base_model.trainable = False
    
    # Agregar capas personalizadas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Crear el modelo final
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compilar el modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def unfreeze_model(model, trainable_layers=50):
    """
    Descongela las últimas capas del modelo para fine-tuning.
    
    Args:
        model: Modelo a descongelar
        trainable_layers: Número de capas a descongelar desde el final
        
    Returns:
        model: Modelo con capas descongeladas
    """
    # Obtener la capa base (ResNet50)
    base_model = model.layers[0]
    
    # Descongelar las últimas capas
    base_model.trainable = True
    
    # Congelar todas las capas excepto las últimas trainable_layers
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    # Recompilar con una tasa de aprendizaje más baja para fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
