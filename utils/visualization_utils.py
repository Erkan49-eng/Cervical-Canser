"""
GÖRSELLEŞTİRME FONKSİYONLARI

Bu modül Grad-CAM, karşılaştırma grafikleri ve diğer
visualizasyonlar için kullanılır.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from config import GRADCAM_DIR, PLOTS_DIR


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Grad-CAM ısı haritası hesaplar.
    
    Grad-CAM (Gradient-weighted Class Activation Mapping):
    - Modelin hangi bölgelere dikkat ettiğini gösterir
    - Sınıflandırma kararlarını yorumlanabilir hale getirir
    
    Parametreler:
    -----------
    img_array : np.ndarray
        Giriş görüntüsü (1, 224, 224, 3)
    model : tf.keras.Model
        Analiz edilecek model
    last_conv_layer_name : str
        Son convolutional katmanın adı
    pred_index : int, optional
        Sınıf indeksi (None ise tahmin edilen sınıf)
    
    Dönüş:
    ------
    np.ndarray : Normalized ısı haritası (0-1)
    """
    
    # Gradient model oluştur
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Gradient tape ile gradyan hesapla
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Son conv katmanın çıktısına göre gradyan hesapla
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Ağırlıklı kombinasyon
    last_conv_layer_output = last_conv_layer_output[0]
    conv_outputs = last_conv_layer_output.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    # Isı haritası oluştur
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    return heatmap


def reverse_colormap(heatmap):
    """
    Isı haritasını renkli formata dönüştürür.
    
    Parametreler:
    -----------
    heatmap : np.ndarray
        Gri ısı haritası (0-1)
    
    Dönüş:
    ------
    np.ndarray : Renkli ısı haritası (RGB)
    """
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(255 - heatmap, cv2.COLORMAP_JET)
    return heatmap


def plot_training_history(history, save_path=None):
    """
    Eğitim ve doğrulama metriklerini çizer.
    
    Parametreler:
    -----------
    history : tf.keras.callbacks.History
        Model eğitim tarihi
    save_path : str, optional
        Grafik kaydetme yolu
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid()
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Val')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid()
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Val')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid()
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'], label='Val')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_models_performance(results_dict, metric='accuracy', save_path=None):
    """
    Modellerin performansını karşılaştıran bar grafik çizer.
    
    Parametreler:
    -----------
    results_dict : dict
        {model_name: metric_value}
    metric : str
        Karşılaştırılacak metrik
    save_path : str, optional
        Grafik kaydetme yolu
    """
    
    models = list(results_dict.keys())
    values = list(results_dict.values())
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(models, values, color='steelblue', alpha=0.8)
    
    # Renk gradyası ekle
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    plt.title(f'Model Performans Karşılaştırması - {metric.upper()}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Değerleri bar üstüne yazma
    for i, (model, value) in enumerate(zip(models, values)):
        plt.text(i, value + 0.02, f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Confusion matrix (Karmaşıklık matrisi) çizer.
    
    Parametreler:
    -----------
    y_true : np.ndarray
        Gerçek etiketler
    y_pred : np.ndarray
        Tahmin edilen etiketler
    class_names : list
        Sınıf adları
    save_path : str, optional
        Grafik kaydetme yolu
    """
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
