# 🧠 Alzheimer's Disease Detection System

An AI-powered web application for detecting Alzheimer's disease stages from brain MRI scans using deep learning and Explainable AI.

## 🎯 Features
- Automated Alzheimer's stage classification (4 classes)
- Explainable AI: Grad-CAM, LIME, SHAP visualizations
- Patient information management
- Multiple image upload (up to 5 images)
- Prediction history tracking
- Real-time predictions (<2 seconds)

## 🏗️ Architecture
- **Model**: EfficientNetB0 + MobileNetV2 (73% accuracy)
- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Explainability**: Grad-CAM, LIME, SHAP

## 📊 Classes
| Class | Description |
|-------|-------------|
| Non Demented | No Alzheimer's detected |
| Very Mild Impairment | Early stage |
| Mild Impairment | Moderate stage |
| Moderate Impairment | Advanced stage |

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/alzheimers-detection-system.git
cd alzheimers-detection-system
```

### 2. Install dependencies
```bash
pip install tensorflow flask flask-cors opencv-python pillow numpy lime shap
```

### 3. Download the Model
> ⚠️ The model file (`alzheimer_128_best.h5`) is not included due to GitHub's file size limit.
> 
> Train your own model using the dataset or contact the repository owner for the model file.

### 4. Run the server
```bash
python server.py
```

### 5. Open in browser
```
http://localhost:5001
```

## 📁 Project Structure
```
├── server.py              # Flask backend
├── index.html             # Web interface
├── script.js              # Frontend logic
├── styles.css             # Styling
├── explainability.py      # LIME & SHAP
├── requirements.txt       # Dependencies
├── metrics_data.json      # Model metrics
└── PROJECT_DOCUMENTATION.md  # Full documentation
```

## 📈 Model Performance
- **Accuracy**: 73%
- **AUC-ROC**: 0.89
- **Input Size**: 128×128 pixels
- **Training Dataset**: 6,400 MRI scans

## 🔬 Explainable AI Methods
- **Grad-CAM**: Visual heatmap showing model attention
- **LIME**: Region-based importance highlighting
- **SHAP**: Quantitative feature contribution analysis

## ⚠️ Disclaimer
This system is a proof-of-concept and is NOT intended for clinical use without proper validation and regulatory approval.

## 👨‍💻 Tech Stack
- Python 3.10.11
- TensorFlow 2.18.0
- Flask
- OpenCV
- LIME & SHAP
