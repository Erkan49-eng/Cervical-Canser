"""
VERİ YÜKLEME VE ÖN İŞLEME FONKSİYONLARI

Bu modül SipakMed veri setini yüklemek, işlemek ve
eğitim/doğrulama/test kümelerine bölmek için kullanılır.
"""

import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import cv2

def load_sipakmed_dataset(data_path, image_size=224, validation_split=0.2, test_split=0.2):
    """
    SipakMed veri setini yükler ve işler.
    
    Parametreler:
    -----------
    data_path : str
        Veri setinin kök dizini
    image_size : int
        Çıktı görüntü boyutu (224x224 önerilir)
    validation_split : float
        Doğrulama set oranı (0.2 = %20)
    test_split : float
        Test set oranı (0.2 = %20)
    
    Dönüş:
    ------
    tuple : (X_train, y_train, X_val, y_val, X_test, y_test)
        Normalleştirilmiş görüntüler ve etiketler
    
    Örnek:
    ------
    >>> X_train, y_train, X_val, y_val, X_test, y_test = \
    ...     load_sipakmed_dataset('/path/to/sipakmed')
    >>> print(X_train.shape)  # (2500, 224, 224, 3)
    """
    
    images = []
    labels = []
    
    # Sınıf klasörlerini oku
    class_dirs = sorted([d for d in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, d))])
    
    print(f"\n📁 Bulunan {len(class_dirs)} sınıf: {class_dirs}\n")
    
    # Her sınıf için görüntüleri yükle
    for class_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"📷 {class_name}: {len(image_files)} görüntü yükleniyor...")
        
        for img_file in tqdm(image_files, desc=f"{class_name}"):
            try:
                # Görüntü yükle
                img_path = os.path.join(class_path, img_file)
                img = load_img(img_path, target_size=(image_size, image_size))
                img_array = img_to_array(img)
                
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"❌ Hata {img_file}: {e}")
    
    # NumPy arrayına dönüştür
    X = np.array(images, dtype='float32')
    y = np.array(labels, dtype='int32')
    
    # Görüntüleri normalleştir (0-1 aralığına)
    X = X / 255.0
    
    print(f"\n✅ Toplam {len(X)} görüntü yüklendi")
    print(f"📊 Veri seti şekli: {X.shape}")
    
    # Train/Val/Test split
    # Önce train+val / test'e böl
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    
    # Train+Val'den train/val'e böl
    val_size_adjusted = validation_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=42, stratify=y_temp
    )
    
    print(f"\n📊 VERI BÖLÜMÜ:")
    print(f"  🔷 Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  🔶 Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  🔴 Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def augment_data(images, labels, augmentation_factor=2):
    """
    Veri seti augmentasyonu (çoğaltma) yapar.
    
    Parametreler:
    -----------
    images : np.ndarray
        Görüntü dizisi (N, H, W, C)
    labels : np.ndarray
        Etiket dizisi
    augmentation_factor : int
        Kaç kez çoğaltılacak
    
    Dönüş:
    ------
    tuple : (augmented_images, augmented_labels)
    """
    augmented_images = []
    augmented_labels = []
    
    for img, label in zip(images, labels):
        # Orijinal görüntü
        augmented_images.append(img)
        augmented_labels.append(label)
        
        # Augmented versiyonlar
        for _ in range(augmentation_factor - 1):
            # Rastgele rotasyon
            angle = np.random.randint(-20, 20)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented_img = cv2.warpAffine(img, M, (w, h))
            
            # Rastgele flip
            if np.random.random() > 0.5:
                augmented_img = cv2.flip(augmented_img, 1)
            
            augmented_images.append(augmented_img)
            augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)


def normalize_images(images, method='minmax'):
    """
    Görüntüleri normalleştirir.
    
    Parametreler:
    -----------
    images : np.ndarray
        Görüntü dizisi
    method : str
        'minmax' veya 'zscore'
    """
    if method == 'minmax':
        return (images - images.min()) / (images.max() - images.min())
    elif method == 'zscore':
        mean = images.mean(axis=(0, 1, 2), keepdims=True)
        std = images.std(axis=(0, 1, 2), keepdims=True)
        return (images - mean) / (std + 1e-6)
    else:
        raise ValueError(f"Bilinmeyen normalizasyon metodu: {method}")
