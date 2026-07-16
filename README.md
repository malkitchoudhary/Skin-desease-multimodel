<p align="center">
  <img src="health_prediction.gif" alt="Project Demo" width="100%">
</p>

# 🩺 Skin Disease Multimodal Classifier

A **multimodal deep learning system** that diagnoses skin lesions by fusing **dermoscopic image data** with **patient clinical metadata** (age, sex, lesion localization) — trained on the **HAM10000** dataset. Instead of relying on visual appearance alone, this model mimics real dermatological reasoning by combining what a lesion *looks like* with *who the patient is*, achieving stronger and more clinically-grounded predictions than single-modality approaches.

---

## 🎯 Why Multimodal?

Most skin cancer classifiers use image-only CNNs. But a dermatologist never diagnoses from a photo alone — a patient's **age**, **sex**, and **where on the body** a lesion appears are all diagnostic signals (e.g., certain lesions are far more common on sun-exposed areas in older patients). This project encodes that intuition directly into the model architecture through a **dual-branch fusion network**.

---

## 🧠 Model Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DATA ACQUISITION                             │
│   HAM10000 (10,015 images) + Metadata (age, sex, localization, dx)   │
└──────────────────────────────────────┬───────────────────────────────┘
                                        │
┌──────────────────────────────────────▼───────────────────────────────┐
│                    PREPROCESSING & EDA                               │
│  • Missing age values imputed (median)                               │
│  • Class distribution, correlation & image-dimension analysis        │
│  • Disease codes mapped to full clinical names                       │
│  • Stratified split → Train / Validation / Test                      │
│  • Age → StandardScaler | Sex, Localization → LabelEncoder           │
└──────────────────────────────────────┬───────────────────────────────┘
                                        │
┌──────────────────────────────────────▼───────────────────────────────┐
│                    TF.DATA PIPELINE                                  │
│  Image path + Tabular vector + Label → decoded, resized (224×224),   │
│  normalized, batched, shuffled, prefetched                           │
└──────────────────────────────────────┬───────────────────────────────┘
                                        │
              ┌─────────────────────────┴─────────────────────────┐
              ▼                                                    ▼
┌───────────────────────────────┐                  ┌───────────────────────────┐
│        IMAGE BRANCH           │                  │       TABULAR BRANCH      │
│  Input: 224×224×3             │                  │  Input: [age, sex, loc]   │
│  → Data Augmentation          │                  │  → Dense(32, relu)        │
│    (flip, rotate, zoom,       │                  │  → Dense(16, relu)        │
│     contrast)                 │                  │                           │
│  → Conv2D(32) + BN + Pool     │                  │                           │
│  → Conv2D(64) + BN + Pool     │                  │                           │
│  → Conv2D(128) + BN + Pool    │                  │                           │
│  → Conv2D(256) + BN + Pool    │                  │                           │
│  → GlobalAveragePooling2D     │                  │                           │
│  → Dense(128, relu)           │                  │                           │
└───────────────┬───────────────┘                  └─────────────┬─────────────┘
                │                                                 │
                └───────────────────┬─────────────────────────────┘
                                     ▼
                         ┌───────────────────────┐
                         │   Concatenate Layer    │  ← Fusion Point
                         └───────────┬────────────┘
                                     ▼
                         Dense(128, relu) → Dropout(0.4)
                                     ▼
                         Dense(7, softmax) → Prediction
└──────────────────────────────────────┬───────────────────────────────┘
                                        │
┌──────────────────────────────────────▼───────────────────────────────┐
│                    TRAINING & EVALUATION                             │
│  Optimizer: Adam | Loss: Sparse Categorical Crossentropy             │
│  Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau        │
│  Metrics: Accuracy, F1-Score, Confusion Matrix, ROC-AUC              │
│  Baseline comparison: 11 classical ML models on tabular data alone   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📌 Project Overview

| | |
|---|---|
| **Problem Type** | Multi-class Image + Tabular Fusion Classification |
| **Dataset** | [HAM10000 — Human Against Machine with 10000 training images](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| **Total Samples** | 10,015 dermoscopic images with paired clinical metadata |
| **Classes** | 7 skin disease categories |
| **Framework** | TensorFlow / Keras (Functional API) |

### Disease Classes

| Code | Disease | Code | Disease |
|------|---------|------|---------|
| `akiec` | Actinic Keratoses | `mel` | Melanoma |
| `bcc` | Basal Cell Carcinoma | `nv` | Melanocytic Nevus |
| `bkl` | Benign Keratosis | `vasc` | Vascular Lesion |
| `df` | Dermatofibroma | | |

---

## 🏗️ Architecture Details

**Fusion strategy:** Late feature-level fusion — each branch independently extracts a feature representation before being concatenated and passed through a shared classification head.

| Branch | Layers | Output |
|---|---|---|
| **Image** | 4× (Conv2D → BatchNorm → MaxPool → Dropout) + GlobalAveragePooling2D + Dense(128) | 128-dim feature vector |
| **Tabular** | Dense(32) → Dense(16) | 16-dim feature vector |
| **Fusion Head** | Concatenate → Dense(128) → Dropout(0.4) → Dense(7, softmax) | 7-class probability distribution |

**Model size:** 443,351 total parameters (442,391 trainable, 960 non-trainable)

---

## 📊 Results

Trained for 25 epochs with early stopping and learning-rate reduction on plateau.

| Metric | Score |
|---|---|
| **Test Accuracy** | **75.39%** |
| **Test Loss** | 0.6431 |
| **Weighted F1-Score** | 0.7359 |
| **Macro F1-Score** | 0.44 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| akiec | 0.33 | 0.26 | 0.29 | 65 |
| bcc | 0.49 | 0.44 | 0.46 | 103 |
| bkl | 0.45 | 0.57 | 0.50 | 220 |
| df | 0.00 | 0.00 | 0.00 | 23 |
| mel | 0.51 | 0.26 | 0.35 | 223 |
| **nv** | **0.87** | **0.93** | **0.90** | 1341 |
| vasc | 0.74 | 0.50 | 0.60 | 28 |

> **Note:** The dataset is heavily imbalanced — `nv` (melanocytic nevus) makes up ~67% of samples, which drives the strong overall accuracy. Minority classes (`akiec`, `df`, `mel`) are harder to detect and are strong candidates for future improvement via class weighting or oversampling.

A separate **baseline comparison** was run using 11 classical ML models (Logistic Regression, Random Forest, XGBoost, CatBoost, etc.) trained on tabular data only, confirming that image information adds substantial discriminative power beyond patient metadata alone.

---

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **Data Handling:** Pandas, NumPy, `tf.data` pipeline
- **Preprocessing:** scikit-learn (LabelEncoder, StandardScaler)
- **Visualization:** Matplotlib, Seaborn
- **Classical ML Baselines:** scikit-learn, XGBoost, CatBoost
- **Dataset Source:** KaggleHub
- **Deployment:** Streamlit

---

## 📁 Project Structure

```
Skin-desease-multimodel/
│
├── Desease_multimodel.ipynb        # Full notebook: EDA → training → evaluation → export
├── saved_model_artifacts/
│   ├── multimodal_skin_model.keras # Trained fusion model
│   ├── label_encoder.pkl           # Disease label encoder
│   ├── sex_encoder.pkl             # Sex encoder
│   ├── localization_encoder.pkl    # Body localization encoder
│   ├── age_scaler.pkl              # Age StandardScaler
│   ├── disease_name_map.pkl        # Short code → full disease name
│   └── app_config.pkl              # Class list & dropdown options
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/malkitchoudhary/Skin-desease-multimodel.git
cd Skin-desease-multimodel
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the app
```bash
streamlit run app.py
```

### 4. Predict
Upload a dermoscopic image, enter patient age, sex, and lesion localization → the model returns the predicted disease with confidence score and full class-probability breakdown.

---

## ⚠️ Disclaimer

This project is built **for educational and portfolio purposes only**. It is **not a certified medical diagnostic tool** and must not be used as a substitute for professional dermatological evaluation. Always consult a qualified medical professional for diagnosis and treatment.

---

## 🔮 Future Improvements

- Address class imbalance with weighted loss / focal loss / oversampling for minority classes (`df`, `akiec`, `mel`)
- Fine-tune with pretrained backbones (EfficientNet, ResNet50) for the image branch
- Add Grad-CAM visualizations for model interpretability
- Expand tabular branch with additional clinical features
- Deploy publicly via Streamlit Cloud / HuggingFace Spaces

---

## 👤 Author

**Malkit Choudhary**
Data Science & Machine Learning Engineer | BCA Student, Manipal University Jaipur

---

## 📄 License

This project is open-source and available for educational use.
