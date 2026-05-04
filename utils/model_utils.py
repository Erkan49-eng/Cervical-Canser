"""
MODEL OLUŞTURMA VE EĞİTİM FONKSİYONLARI

Bu modül Transfer Learning modellerini oluşturmak,
eğitmek ve değerlendirmek için kullanılır.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, ResNet101, ResNet152,
    EfficientNetB0, EfficientNetB3, EfficientNetB5,
    MobileNetV2, MobileNetV3Large,
    DenseNet121, DenseNet169,
    VGG16,
    InceptionV3,
    Xception
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from config import NUM_CLASSES, LEARNING_RATE, DROPOUT_RATE, L2_REGULARIZATION


def get_pretrained_model(model_name, num_classes=NUM_CLASSES, input_shape=(224, 224, 3)):
    """
    Pre-trained Transfer Learning modeli oluşturur.
    
    Parametreler:
    -----------
    model_name : str
        Model adı ('resnet50', 'efficientnetb0', vs.)
    num_classes : int
        Çıktı sınıf sayısı (7 rahim kanseri sınıfı)
    input_shape : tuple
        Giriş görüntü şekli
    
    Dönüş:
    ------
    tf.keras.Model : Derlenmemiş model
    
    Nasıl çalışır:
    ---------------
    1. ImageNet pre-trained ağırlıkları ile base model yüklenir
    2. Base model dondurulur (transfer learning)
    3. Global Average Pooling katmanı eklenir
    4. Dropout katmanı eklenir (overfitting önleme)
    5. Dense katman eklenir
    6. Çıktı katmanı (7 sınıf için Softmax)
    
    Örnek:
    ------
    >>> model = get_pretrained_model('resnet50', num_classes=7)
    >>> model.summary()
    """
    
    # Model seçimi
    model_dict = {
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152,
        'efficientnetb0': EfficientNetB0,
        'efficientnetb3': EfficientNetB3,
        'efficientnetb5': EfficientNetB5,
        'mobilenetv2': MobileNetV2,
        'mobilenetv3large': MobileNetV3Large,
        'densenet121': DenseNet121,
        'densenet169': DenseNet169,
        'vgg16': VGG16,
        'inceptionv3': InceptionV3,
        'xception': Xception
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Bilinmeyen model: {model_name}")
    
    # Base model yükle (ImageNet ağırlıkları ile)
    base_model = model_dict[model_name](input_shape=input_shape, 
                                        weights='imagenet', 
                                        include_top=False)
    
    # Base modeli dondur (transfer learning)
    base_model.trainable = False
    
    # Yeni model oluştur
    model = models.Sequential([
        # Giriş
        layers.Input(shape=input_shape),
        
        # Base model (pre-trained)
        base_model,
        
        # Global Average Pooling (boyut azaltma)
        layers.GlobalAveragePooling2D(),
        
        # Dense katman
        layers.Dense(512, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
        layers.Dropout(DROPOUT_RATE),
        
        # Dense katman
        layers.Dense(256, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
        layers.Dropout(DROPOUT_RATE),
        
        # Çıktı katmanı (sınıf sayısı kadar)
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Modeli derler (eğitim için hazırlar).
    
    Parametreler:
    -----------
    model : tf.keras.Model
        Derlenmek istenen model
    learning_rate : float
        Optimizer öğrenme hızı
    
    Dönüş:
    ------
    tf.keras.Model : Derlenmiş model
    """
    
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC()]
    )
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, 
               epochs=30, batch_size=32, verbose=1):
    """
    Modeli eğitir.
    
    Parametreler:
    -----------
    model : tf.keras.Model
        Derlenmiş model
    X_train, y_train : np.ndarray
        Eğitim verisi
    X_val, y_val : np.ndarray
        Doğrulama verisi
    epochs : int
        Eğitim dönem sayısı
    batch_size : int
        Batch boyutu
    verbose : int
        Çıktı seviyesi (0=sessiz, 1=normal, 2=detaylı)
    
    Dönüş:
    ------
    tf.keras.callbacks.History : Eğitim tarihi
    """
    
    # Early Stopping (eğitim sonlandırma)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Öğrenme hızı azaltma
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    # Eğitim
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose
    )
    
    return history


def evaluate_model(model, X_test, y_test, batch_size=32):
    """
    Modeli test verisi üzerinde değerlendirir.
    
    Parametreler:
    -----------
    model : tf.keras.Model
        Değerlendirilecek model
    X_test, y_test : np.ndarray
        Test verisi
    batch_size : int
        Batch boyutu
    
    Dönüş:
    ------
    dict : Performans metrikleri
    """
    
    results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    
    metrics_names = model.metrics_names
    metrics = {name: value for name, value in zip(metrics_names, results)}
    
    return metrics


def get_model_size(model):
    """
    Modelin boyutunu hesaplar (MB cinsinden).
    
    Parametreler:
    -----------
    model : tf.keras.Model
        Boyutu hesaplanacak model
    
    Dönüş:
    ------
    float : Model boyutu (MB)
    """
    
    total_params = model.count_params()
    # Her parametre 4 byte (float32)
    model_size_mb = (total_params * 4) / (1024 * 1024)
    
    return model_size_mb


def unfreeze_base_model(model, unfreeze_from_layer=-1):
    """
    Base modeli (kısmen) çözer (fine-tuning için).
    
    Parametreler:
    -----------
    model : tf.keras.Model
        Model
    unfreeze_from_layer : int
        Hangi katmandan itibaren çözülecek (-1 = en son katman)
    
    Dönüş:
    ------
    tf.keras.Model : Güncellenmiş model
    """
    
    # Base modeli bul
    base_model = model.layers[2]  # Sequential modelde 3. layer (index 2)
    
    # Son katmanlarını çöz
    for layer in base_model.layers[unfreeze_from_layer:]:
        layer.trainable = True
    
    return model
