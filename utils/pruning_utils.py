"""
PRUNING (BUDAMA) FONKSİYONLARI

Bu modül ağırlık budama ve yapısal budama tekniklerini
uygulamak için kullanılır.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow_model_optimization as tfmot


def calculate_layer_sparsity(layer):
    """
    Bir katmanın sparsity'sini (seyreklik) hesaplar.
    
    Sparsity = Sıfır olan ağırlıklar / Toplam ağırlıklar
    
    Parametreler:
    -----------
    layer : tf.keras.layers.Layer
        Analiz edilecek katman
    
    Dönüş:
    ------
    float : Sparsity değeri (0-1 arası)
    
    Örnek:
    ------
    >>> layer = model.layers[5]
    >>> sparsity_val = calculate_layer_sparsity(layer)
    >>> print(f"Sparsity: {sparsity_val:.2%}")  # Sparsity: 15.42%
    """
    
    if not layer.weights:
        return 0.0
    
    weights = layer.weights[0].numpy()
    zero_count = np.sum(weights == 0)
    total_count = weights.size
    
    sparsity_value = zero_count / total_count if total_count > 0 else 0
    return sparsity_value


def analyze_layer_variance(model, X_sample, layer_index=None):
    """
    Modelin katmanlarının varyansını analiz eder (Hocamın yöntemi).
    
    Bu yöntem her katmanın çıktısının varyansını hesaplar.
    Düşük varyans = Katman az bilgi işliyor = Budanabilir
    
    Parametreler:
    -----------
    model : tf.keras.Model
        Analiz edilecek model
    X_sample : np.ndarray
        Tek bir görüntü veya az sayıda görüntü
    layer_index : int, optional
        Spesifik katman indeksi (None ise tüm katmanlar)
    
    Dönüş:
    ------
    dict : {layer_name: variance_value}
    
    Örnek:
    ------
    >>> X_sample = X_test[:1]  # İlk test görüntüsü
    >>> variances = analyze_layer_variance(model, X_sample)
    >>> min_var_layer = min(variances, key=variances.get)
    >>> print(f"En düşük varyans: {min_var_layer}")
    """
    
    # Her katmanın çıktısını alacak activation model oluştur
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    
    # Tahminleri al
    activations = activation_model.predict(X_sample, verbose=0)
    
    # Varyansları hesapla
    variances = {}
    for i, layer in enumerate(model.layers):
        if isinstance(activations[i], np.ndarray):
            var = np.var(activations[i])
            variances[layer.name] = var
    
    return variances


def apply_unstructured_pruning(model, sparsity_target=0.5):
    """
    Unstructured Pruning (Ağırlık Budama) uygular.
    
    Mantık:
    ------
    1. En düşük önem seviyesine sahip ağırlıkları belirle
    2. Bu ağırlıkları sıfır yap
    3. Model daha seyrek (sparse) hale gelir
    4. Sıfır ağırlıklar atlanabilir → daha hızlı çıkarım
    
    Parametreler:
    -----------
    model : tf.keras.Model
        Budanacak model
    sparsity_target : float
        Hedef sparsity (0.5 = %50 ağırlık sıfırlansın)
    
    Dönüş:
    ------
    tf.keras.Model : Budanmış model
    
    Örnek:
    ------
    >>> pruned_model = apply_unstructured_pruning(model, sparsity_target=0.7)
    >>> print(f"Sparsity: %70")
    """
    
    # Pruning yapılandırması
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=sparsity_target,
        begin_step=0,
        end_step=1000
    )
    
    # Pruning'i modele uygula
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        model,
        pruning_schedule=pruning_schedule
    )
    
    return pruned_model


def apply_structured_pruning(model, pruning_rate=0.3):
    """
    Structured Pruning (Kanal Budama) uygular.
    
    Mantık:
    ------
    1. Tüm çıkış filtrelerini kaldır (kanal budama)
    2. Model boyutu daha fazla azalır
    3. Daha hızlı çıkarım (GPU dostu)
    
    Parametreler:
    -----------
    model : tf.keras.Model
        Budanacak model
    pruning_rate : float
        Budanacak kanal oranı (0.3 = %30 kanal silinir)
    
    Dönüş:
    ------
    tf.keras.Model : Budanmış model
    """
    
    new_model = models.Sequential()
    
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            # Kanal sayısını azalt
            new_filters = max(1, int(layer.filters * (1 - pruning_rate)))
            new_layer = layers.Conv2D(
                filters=new_filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                activation=layer.activation
            )
            new_model.add(new_layer)
        else:
            new_model.add(layer)
    
    return new_model


def analyze_sparsity_by_layer(model):
    """
    Modelin her katmanının sparsity'sini analiz eder.
    
    Dönüş:
    ------
    dict : Katman adı -> Sparsity değeri
    """
    
    sparsities = {}
    
    for layer in model.layers:
        if layer.weights:
            sparsity_val = calculate_layer_sparsity(layer)
            sparsities[layer.name] = sparsity_val
    
    return sparsities


def find_optimal_cut_layer(model, X_sample):
    """
    Hocamın yaklaşımı: En düşük sparsity'li katmanı bulur.
    
    Bu katmandan model kesilir ve yeniden eğitilir.
    
    Dönüş:
    ------
    str : Kesme noktası (katman adı)
    int : Katman indeksi
    """
    
    variances = analyze_layer_variance(model, X_sample)
    
    # En düşük varyansa sahip katmanı bul
    min_var_layer = min(variances, key=variances.get)
    
    # Katman indeksini bul
    for idx, layer in enumerate(model.layers):
        if layer.name == min_var_layer:
            return min_var_layer, idx
    
    return None, None


def cut_model_at_layer(model, layer_index):
    """
    Modeli belirtilen katmandan keser.
    
    Parametreler:
    -----------
    model : tf.keras.Model
        Kesilecek model
    layer_index : int
        Kesme noktası (bu katmandan sonra kesilir)
    
    Dönüş:
    ------
    tf.keras.Model : Kesilmiş model
    """
    
    # Yeni çıktı katmanı
    new_output = model.layers[layer_index].output
    
    # Yeni model oluştur
    pruned_model = models.Model(inputs=model.input, outputs=new_output)
    
    # Sınıflandırma katmanları ekle
    x = layers.GlobalAveragePooling2D()(new_output)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(7, activation='softmax')(x)
    
    final_model = models.Model(inputs=model.input, outputs=output)
    
    return final_model
