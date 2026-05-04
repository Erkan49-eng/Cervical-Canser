# ============================================
# PROJE KONFİGÜRASYONU
# ============================================

import os

# ============================================
# VERİ AYARLARI
# ============================================

DATA_PATH = '/kaggle/input/sipakmed'  # Kaggle'da veri set yolu
IMAGE_SIZE = 224  # Görüntü boyutu (piksel)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2

# Sınıf eşleştirmesi
CLASS_LABELS = {
    0: 'nv',      # Nevus (Benign)
    1: 'mel',     # Melanoma (Kanserli)
    2: 'bkl',     # Benign Keratosis
    3: 'bcc',     # Basal Cell Carcinoma
    4: 'akiec',   # Actinic Keratosis
    5: 'vasc',    # Vascular
    6: 'df'       # Dermatofibroma
}

NUM_CLASSES = len(CLASS_LABELS)

# ============================================
# MODEL AYARLARI
# ============================================

# 13 Transfer Learning Modeli
MODELS_TO_TRAIN = [
    'resnet50',
    'resnet101',
    'resnet152',
    'efficientnetb0',
    'efficientnetb3',
    'efficientnetb5',
    'mobilenetv2',
    'mobilenetv3large',
    'densenet121',
    'densenet169',
    'vgg16',
    'inceptionv3',
    'xception'
]

# Eğitim parametreleri
EPOCHS_BASELINE = 30
EPOCHS_FINETUNING = 15
EPOCHS_PRUNING = 10
LEARNING_RATE = 0.001
LEARNING_RATE_FINETUNING = 0.0001

# ============================================
# PRUNING AYARLARI
# ============================================

# Unstructured Pruning sparsity oranları
PRUNING_SPARSITY_RATIOS = [0.5, 0.7, 0.9]  # %50, %70, %90

# Structured Pruning kanal kesme oranları
STRUCTURED_PRUNING_RATIOS = [0.3, 0.5]  # %30, %50

# ============================================
# GRADIENT CLIPPING ve REGULARIZATION
# ============================================

GRADIENT_CLIP_VALUE = 1.0
L2_REGULARIZATION = 1e-4
DROPOUT_RATE = 0.5

# ============================================
# ÇIKTI DİZİNLERİ
# ============================================

RESULTS_DIR = 'results'
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')
GRADCAM_DIR = os.path.join(RESULTS_DIR, 'gradcam')

# Dizinleri oluştur
for dir_path in [MODELS_DIR, PLOTS_DIR, REPORTS_DIR, GRADCAM_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================
# RANDOM SEED (Tekrarlanabilirlik)
# ============================================

RANDOM_SEED = 42

# ============================================
# GRAD-CAM AYARLARI
# ============================================

# Her modelin son convolutional katmanı
LAST_CONV_LAYERS = {
    'resnet50': 'conv5_block3_out',
    'resnet101': 'conv5_block3_out',
    'resnet152': 'conv5_block3_out',
    'efficientnetb0': 'top_activation',
    'efficientnetb3': 'top_activation',
    'efficientnetb5': 'top_activation',
    'mobilenetv2': 'out_relu',
    'mobilenetv3large': 'top_activation',
    'densenet121': 'relu',
    'densenet169': 'relu',
    'vgg16': 'block5_conv3',
    'inceptionv3': 'mixed10',
    'xception': 'block14_sepconv2_act'
}

# ============================================
# PERFORMANS METRİKLERİ
# ============================================

METRICS_TO_TRACK = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'auc',
    'model_size_mb',
    'inference_time_ms',
    'parameter_count'
]
