# Facial Emotion Recognition Using Facial Landmarks and Machine Learning

## 👥 Team Members
-   Alexander Nicholas
-   Kieran Hawkins
-   Michael Duggan

## 📌 Project Overview
This project focuses on building a machine learning system that recognizes **facial emotions** using only **facial landmark-based features**—no deep CNNs or raw image classification. We use a mix of **interpretable models** and one **neural network** for comparison.

The primary goal is to assess whether lightweight models can classify emotions effectively using geometric patterns in facial expressions.

---

## 🧠 Models Used
Each team member was responsible for one model:
- **Logistic Regression** () – baseline linear model for interpretability.
- **Decision Tree** () – visual and rule-based approach for capturing nonlinear relations.
- **Multilayer Perceptron (MLP)** () – simple neural network for benchmarking performance.

---

## 📂 Dataset
- **Source**: [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- 35,000+ 48x48 grayscale facial images labeled with 7 emotion categories:
  - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

---

## 🔧 Preprocessing Pipeline
1. **Face Detection** – Detect faces using `face_recognition`.
2. **Landmark Extraction** – 68 facial landmarks per image.
3. **Feature Engineering** – Distances, angles, ratios between key points.
4. **Standardization** – Normalize features across dataset.

---

## 🧪 Experiments
We evaluated each model using:
- Train/Test split (80/20)
- Cross-validation (5-fold)
- Metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

Hyperparameter...

---

## 📊 Results Summary


---

## 📚 Libraries Used
- `face_recognition`
- `opencv-python`
- `numpy`, `pandas`
- `scikit-learn`
- `matplotlib`, `seaborn`

Optional for MLP:
- `keras` or `sklearn.neural_network.MLPClassifier`

---

## 🧾 How to Run
1. Clone the repo or open the Colab notebook.
2. Install dependencies:
   ```bash
   pip install face_recognition opencv-python scikit-learn
