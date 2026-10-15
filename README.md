# EcoSort AI - Deep Learning Waste Classifier

An end-to-end intelligent waste image classification system built with **TensorFlow 2.x**, **Keras 3**, **MobileNetV2 Transfer Learning**, and **Streamlit**.

---

## ♻️ Waste Categories (6 Classes)
The model classifies waste into 6 distinct categories:
- **Cardboard** (Boxes, packaging)
- **Glass** (Bottles, jars)
- **Metal** (Aluminum and tin food/beverage cans)
- **Paper** (Clean paper, newspapers, magazines)
- **Plastic** (Bottles, containers)
- **Trash** (Non-recyclable general household waste)

---

## 📂 Project Architecture

```
DeepLearning/
├── data/
│   └── raw/
│       ├── cardboard/
│       ├── glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       └── trash/
├── models/
│   ├── best_waste_classifier.keras
│   ├── final_waste_classifier.keras
│   ├── class_names.json
│   └── training_results.json
├── outputs/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── evaluation_metrics.json
├── src/
│   ├── preprocess.py      # tf.data pipeline with caching & deterministic splits
│   ├── train.py           # Two-phase transfer learning & fine-tuning
│   ├── evaluate.py        # Fast vectorized evaluation on test set
│   ├── predict.py         # CLI image prediction tool with probability rankings
│   └── app.py             # Streamlit web application with session state & camera support
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   ```
   *Windows (PowerShell):*
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

All commands should be executed from the project root (`d:\DeepLearning`):

### 1. Train Model
Trains the model using a 2-stage transfer learning approach (Phase 1: frozen MobileNetV2 head training; Phase 2: fine-tuning top 30 layers with lower learning rate):
```bash
python src/train.py
```

### 2. Evaluate Performance
Evaluates the best checkpoint on the held-out test split, saving confusion matrix and classification report:
```bash
python src/evaluate.py
```

### 3. Predict Single Image (CLI)
Classifies a local image file and displays ranked class probabilities:
```bash
python src/predict.py --image path/to/sample.jpg
```

### 4. Launch Web Application
Starts the interactive "EcoSort AI" Streamlit interface (supports both image upload and live camera capture):
```bash
streamlit run src/app.py
```

---

## 📊 Model & Training Details
- **Backbone:** MobileNetV2 pretrained on ImageNet
- **Input Dimensions:** $224 \times 224 \times 3$
- **Data Augmentation:** Random Flip, Rotation, Zoom, and Contrast
- **Normalization:** Embedded in the computational graph (raw $[0, 255]$ inputs)
- **Class Balancing:** Balanced class weighting via Scikit-Learn
- **Callbacks:** ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

