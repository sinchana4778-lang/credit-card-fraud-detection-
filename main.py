# =========================================
# CREDIT CARD FRAUD DETECTION (FINAL VERSION)
# =========================================

# 1. IMPORTS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# =========================================
# 2. LOAD DATA
# =========================================

file_path = os.path.join("data", "creditcard.csv")

if not os.path.exists(file_path):
    print("❌ Dataset not found! Put 'creditcard.csv' inside data/ folder.")
    exit()

df = pd.read_csv(file_path)

print("\n✅ Dataset Loaded Successfully!\n")
print(df.head())

# =========================================
# 3. CREATE FOLDERS
# =========================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =========================================
# 4. EDA - CLASS DISTRIBUTION
# =========================================

print("\n🔍 Class Distribution:\n")
print(df['Class'].value_counts())

plt.figure()
df['Class'].value_counts().plot(kind='bar')
plt.title("Class Distribution (Before SMOTE)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("outputs/class_distribution.png")
plt.close()

# =========================================
# 5. PREPROCESSING
# =========================================

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================
# 6. TRAIN-TEST SPLIT (BEFORE SMOTE) ✅
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================
# 7. APPLY SMOTE ONLY ON TRAINING DATA ✅
# =========================================

print("\n⚙️ Applying SMOTE on training data...\n")

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

print("After SMOTE (Training Set):")
print(pd.Series(y_train).value_counts())

# =========================================
# 8. MODEL TRAINING (FAST + BALANCED)
# =========================================

print("\n🚀 Training Model...\n")

model = RandomForestClassifier(
    n_estimators=20,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

print("✅ Model training completed")

# =========================================
# 9. PREDICTION
# =========================================

probs = model.predict_proba(X_test)[:, 1]
y_pred = (probs > 0.7).astype(int)  # try 0.6–0.8
probs = model.predict_proba(X_test)[:, 1]

print("\n🔢 Sample Fraud Probability:", probs[0])
print(f"Fraud Probability: {probs[0]:.4f}")
if probs[0] > 0.7:
    print("🚨 High Risk Fraud!")
elif probs[0] > 0.4:
    print("⚠️ Medium Risk")
else:
    print("✅ Low Risk")

# =========================================
# 10. EVALUATION
# =========================================

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# =========================================
# 11. SAMPLE PREDICTION
# =========================================

print("\n🔍 Testing Sample Transaction...\n")

sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

if prediction[0] == 1:
    print("🚨 Fraud Detected!")
else:
    print("✅ Normal Transaction")

# =========================================
# 12. SAVE MODEL
# =========================================

joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n💾 Model saved in 'models/fraud_model.pkl'")

# =========================================
# DONE
# =========================================

print("\n🎉 PROJECT COMPLETED SUCCESSFULLY!")