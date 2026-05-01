# 💳 Credit Card Fraud Detection System

## 🚀 Live Demo
👉 https://hknvh4hx8endfbbune4mtq.streamlit.app/

---

## 📌 Project Overview
This project is a Machine Learning-based Credit Card Fraud Detection System that identifies fraudulent transactions using real-world imbalanced classification techniques. It predicts whether a transaction is fraudulent or normal and provides a risk score.

---

## 🎯 Objective
The goal of this project is to build an intelligent system that can:
- Detect fraudulent transactions in financial data
- Reduce financial risk for users and institutions
- Provide fraud probability scoring
- Classify transactions into Low, Medium, and High Risk

---

## 🧠 Machine Learning Workflow
The system follows a complete ML pipeline:

Data Collection → Data Preprocessing → Feature Scaling → Handling Imbalanced Data (SMOTE) → Model Training → Evaluation → Prediction → Deployment

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Streamlit (for Web App)
- Joblib (Model Saving)

---

## 📊 Model Performance
- High Recall for Fraud Detection (~0.79–0.87 depending on threshold)
- Balanced Precision-Recall Trade-off
- Accuracy ~99% (on test data)
- Optimized for real-world fraud detection scenario

---

## ⚙️ Features
- Real-time fraud prediction
- Fraud probability score
- Risk classification system:
  - 🟢 Low Risk Transaction
  - 🟡 Medium Risk Transaction
  - 🔴 High Risk Fraud
- Interactive Streamlit dashboard
- User-friendly interface

---

## 📁 Project Structure
Credit-Card-Fraud-Detection/
│
├── app.py
├── main.py
├── models/
│   ├── fraud_model.pkl
│   └── scaler.pkl
├── outputs/
│   ├── class_distribution.png
│   └── confusion_matrix.png
├── requirements.txt
└── README.md

---

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

📌 Dataset

Dataset used:
Credit Card Fraud Detection Dataset (Kaggle)

Note: Dataset is not included in this repository due to GitHub size limitations.

🔥 Key Learnings

- Handling highly imbalanced datasets using SMOTE
- Building end-to-end ML pipeline
- Feature scaling and preprocessing techniques
-  Model evaluation using precision, recall, and F1-score
-  Deploying ML models using Streamlit Cloud

🚀 Deployment

The model is deployed using Streamlit Cloud and is accessible at:
👉 https://hknvh4hx8endfbbune4mtq.streamlit.app/

👨‍💻 Author

Sinchana Gowda

⭐ If you like this project

Give this repository a star ⭐ and connect with me on LinkedIn.