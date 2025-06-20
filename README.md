# 🧠 Parkinson's Disease Detection using AI and Sensor Data

This project presents a machine learning-powered web application for the **early detection of Parkinson’s Disease (PD)** using motion sensor data and questionnaire responses. The system aims to assist healthcare providers and researchers in identifying early signs of PD, especially in remote or under-resourced settings.

---

## 🔍 Overview

Parkinson’s Disease is a progressive neurological disorder that is often diagnosed too late. This project proposes an AI-based solution to:
- Detect tremors and motor dysfunctions using wearable sensor data.
- Differentiate between **Healthy**, **Parkinson's Disease (PD)**, and **Differential Diagnosis (DD)** conditions.
- Provide a real-time, user-friendly diagnostic web interface.

---

## 🛠️ Features

- 📊 **High-dimensional feature engineering** from accelerometer and gyroscope signals (66 features per patient).
- ⚙️ **Ensemble learning** with models like Random Forest, XGBoost, CatBoost, and Extra Trees.
- 🧪 **Advanced feature selection** using:
  - Permutation Feature Importance (PFI)
  - Recursive Feature Elimination (RFE)
  - SelectKBest
  - Mutual Information
- ⚖️ **Balanced dataset** using SMOTE to handle class imbalance.
- 🌐 **Flask-based web app** for real-time predictions and user inputs.
- 📉 Reduced testing effort using just **2 optimized movement tasks** per classification.

---

## 🧬 Dataset

We used the **PADS (Parkinson’s Disease Smartwatch)** dataset containing:
- Sensor data collected from 469 patients performing 11 predefined movement tasks using wrist-worn smartwatches.
- Questionnaire-based non-motor symptoms and demographic information.
- 3-class labels: **Healthy**, **PD**, and **DD**.

---

## 🚀 Model Highlights

| Task                         | Accuracy | ROC-AUC |
|-----------------------------|----------|---------|
| Healthy vs Unhealthy        | 92%      | 0.94    |
| Parkinson’s vs DD           | 85%      | 0.86    |
| Multiclass (HC vs PD vs DD) | 81%      | —       |

- Final model: **Soft Voting Ensemble**
- Top movement tasks: `StretchHold`, `CrossArms`, `DrinkGlass`, `HoldWeight`

---

## 🧑‍💻 Technologies Used

- Python
- Flask
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- SMOTE (imbalanced-learn)
- HTML, CSS (for frontend forms)

---
