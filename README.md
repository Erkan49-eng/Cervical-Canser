# Rahim Ağzı Kanseri Görüntü Sınıflandırması ve Model Optimizasyonu

## 📋 Proje Özeti

Bu proje, **Transfer Learning** kullanarak rahim ağzı kanseri görüntülerini sınıflandırmakta ve **Pruning** tekniği ile modelleri optimize etmektedir.

### 🎯 Hedefler
- 13 farklı pre-trained modeli SipakMed veri seti üzerinde eğitmek
- Her modele Unstructured ve Structured Pruning uygulamak
- Parametre sayısını azaltırken performansı koruyan optimal modelleri bulmak
- Grad-CAM ile model kararlarını görselleştirmek

---

## 📊 Veri Seti

**SipakMed Veri Seti (4049 Görüntü)**
- **7 Sınıf**: nv, mel, bkl, bcc, akiec, vasc, df
- **Kaynak**: Kaggle
- **Görüntü Boyutu**: 224x224 RGB

---

## 🧠 Kullanılan Modeller (13 Adet)

1. ResNet50
2. ResNet101
3. ResNet152
4. EfficientNetB0
5. EfficientNetB3
6. EfficientNetB5
7. MobileNetV2
8. MobileNetV3Large
9. DenseNet121
10. DenseNet169
11. VGG16
12. InceptionV3
13. Xception

---

## 🔧 Parametre Azaltma Teknikleri

### 1. **Unstructured Pruning (Ağırlık Budama)**
- Düşük önem seviyesine sahip ağırlıkları sıfırla
- Sparsity oranı: %50, %70, %90

### 2. **Structured Pruning (Kanal Budama)**
- Tüm çıkış filtrelerini kaldır
- Daha düşük model boyutu ve hızlı çıkarım

### 3. **Sparsity Analizi (Hocamın Yaklaşımı)**
- Her katmanın varyansını hesapla
- En düşük sparsity'li katmanı belirle
- Modeli bu noktadan kes
- Fine-tuning ile yeniden eğit

---

## 📁 Proje Yapısı

```
Cervical-Canser/
├── config.py                          # Konfigürasyonlar
├── requirements.txt                   # Kütüphaneler
├── README.md                          # Bu dosya
├── utils/
│   ├── __init__.py
│   ├── data_utils.py                 # Veri işleme
│   ├── model_utils.py                # Model oluşturma
│   ├── pruning_utils.py              # Pruning işlevleri
│   └── visualization_utils.py        # Görselleştirme
├── notebooks/
│   ├── 01_data_preparation.ipynb     # Veri hazırlama
│   ├── 02_baseline_training.ipynb    # 13 modeli eğitme
│   ├── 03_pruning_optimization.ipynb # Pruning uygulama
│   └── 04_comparison_analysis.ipynb  # Karşılaştırma
└── results/
    ├── models/                       # Eğitilmiş modeller
    ├── plots/                        # Grafikler
    ├── reports/                      # Raporlar
    └── gradcam/                      # Grad-CAM görselleri
```

---

## 🚀 Hızlı Başlangıç (Kaggle'da)

### Adım 1: Kütüphaneleri Yükle
```bash
!pip install -r requirements.txt
```

### Adım 2: Veri Setini Bağla
Kaggle Notebook'ta **Data** sekmesinden SipakMed veri setini ekle.

### Adım 3: Notebook'ları Sırasıyla Çalıştır
```
01_data_preparation.ipynb      (Veri yükleme ve ön işleme)
    ↓
02_baseline_training.ipynb     (13 modeli eğitme)
    ↓
03_pruning_optimization.ipynb  (Pruning uygulama)
    ↓
04_comparison_analysis.ipynb   (Karşılaştırma ve analiz)
```

---

## 📊 Beklenen Sonuçlar

| Aşama | Hedef | Beklenen Sonuç |
|-------|-------|----------------|
| Baseline | Orijinal model performansı | ~90-95% Accuracy |
| Unstructured Pruning (%70) | Model boyutu 70% azalt | ~88-93% Accuracy |
| Structured Pruning | Model boyutu 50-60% azalt | ~85-92% Accuracy |
| Sparsity Analizi | Optimal kesme noktası | ~87-92% Accuracy |
| Fine-tuning | Performans recovery | Original'e yakın |

---

## 📈 Çıktılar

1. **Model Performans Raporu** (CSV)
2. **Karşılaştırma Grafikleri** (PNG)
3. **Grad-CAM Görselleştirmeleri**
4. **Pruning Analiz Raporu**
5. **Optimized Models** (SavedModel format)

---

## 🔬 Kod Örnekleri

### Model Eğitme
```python
from utils.model_utils import get_pretrained_model
from utils.data_utils import load_sipakmed_dataset

# Veri yükleme
X_train, y_train, X_val, y_val, X_test, y_test = load_sipakmed_dataset('/path/to/data')

# Model oluşturma
model = get_pretrained_model('resnet50', num_classes=7)

# Eğitme
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30)
```

### Pruning Uygulama
```python
from utils.pruning_utils import apply_unstructured_pruning, fine_tune_model

# Pruning
pruned_model = apply_unstructured_pruning(model, sparsity=0.7)

# Fine-tuning
fine_tuned_model = fine_tune_model(pruned_model, X_train, y_train, epochs=10)
```

---

## 👨‍🏫 Danışman

Proje Danışmanı: **Dr. Öğr. Üyesi Yahya DOĞAN** (Siirt Üniversitesi)

---

## 📝 Notlar

- Tüm modeller ImageNet pre-trained ağırlıkları ile başlar
- Transfer learning sayesinde daha az veri ile başarılı sonuçlar alınır
- Pruning sonrasında fine-tuning kritik önem taşır
- GPU kullanımı önerilir (Kaggle GPU tersiz sunucu kullanılabilir)

---

## 📞 İletişim

**GitHub**: [@Erkan49-eng](https://github.com/Erkan49-eng)

---

**Son Güncelleme**: 2026-05-04
