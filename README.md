# 🧠 Oral Cancer Detection using Deep Learning & Explainable AI (XAI)

## 📌 Project Overview

This project focuses on detecting **Oral Cancer** from medical images using **Machine Learning and Deep Learning techniques**. It combines traditional algorithms like **SVM** with a **custom ResNet50-based model**, along with **Explainable AI (XAI)** using LIME to interpret predictions.

The system provides:

* Cancer prediction from input images
* Model performance comparison
* Visual explanation of predictions
* Basic healthcare precaution suggestions

---

## 🚀 Key Features

* 🖼️ Image-based Oral Cancer Prediction
* 🧠 Deep Learning using ResNet50 architecture
* 📊 Machine Learning using SVM
* 📈 Performance Metrics (Accuracy, Precision, Recall, F1-Score)
* 🔍 Explainable AI using LIME (visual explanation)
* 📉 Confusion Matrix & Graph Visualization
* 💡 Health Precaution Suggestions
* 🐳 Docker Support for easy environment setup

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Libraries & Frameworks:**

  * NumPy, Pandas
  * OpenCV, PIL
  * Scikit-learn
  * TensorFlow / Keras
  * Matplotlib, Seaborn
  * Scikit-image
  * LIME (Explainable AI)
* **GUI:** Tkinter
* **Containerization:** Docker

---

## 🧪 Machine Learning Models

### 1. Support Vector Machine (SVM)

* Used for classification based on extracted features
* Provides baseline comparison

### 2. Proposed ResNet50 Model

* Deep learning model for improved accuracy
* Custom architecture using convolution layers
* Trained on extracted image features

---

## 🔍 Feature Extraction Techniques

The model uses hybrid feature extraction:

* 🎨 Color Histogram Features
* 🧩 Texture Features (GLCM - Gray Level Co-occurrence Matrix)
* 🌿 Green Pixel Removal for noise reduction
* 📊 Statistical Features (Mean, Std, Entropy, Skewness, Kurtosis)

---

## 📊 Performance Metrics

The following metrics are used for evaluation:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix Visualization

---

## 🔬 Explainable AI (XAI)

* Implemented using **LIME (Local Interpretable Model-Agnostic Explanations)**
* Highlights important regions of the image influencing prediction
* Helps in model transparency and trust

---

## 🖥️ How to Run the Project

### ▶️ Option 1: Run Normally (Python)

```bash
pip install -r requirements.txt
python Main.py
```

---

### 🐳 Option 2: Run Using Docker

#### Step 1: Build Docker Image

```bash
docker build -t oral-cancer-app .
```

#### Step 2: Run Container

```bash
docker run -it oral-cancer-app
```

---


---

## 📸 Workflow

1. Upload Dataset
2. Extract Features
3. Train Models (SVM & ResNet50)
4. Evaluate Performance
5. Predict using test images
6. Generate XAI explanation (LIME)


---

## 👩‍💻 Author

**Likhitha Sivapuram**

---

## 📜 License

This project is for academic and research purposes only.
