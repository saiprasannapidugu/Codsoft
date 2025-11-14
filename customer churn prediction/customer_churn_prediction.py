# ==========================================================
# CUSTOMER CHURN PREDICTION
# Algorithms: Logistic Regression, Random Forest, Gradient Boosting
# Dataset: Customer_Churn_Dataset.csv
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================================
# 1. Load Data
# ==========================================================
df = pd.read_csv("Customer_Churn_Dataset.csv")
print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head(), "\n")

# ==========================================================
# 2. Preprocessing
# ==========================================================
# Drop unnecessary columns
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Encode categorical variables
label_enc = LabelEncoder()
df["Gender"] = label_enc.fit_transform(df["Gender"])  # Male=1, Female=0
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

# Split features & target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("✅ Data preprocessing complete.\n")

# ==========================================================
# 3. Model Training and Evaluation
# ==========================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# ==========================================================
# 4. Compare Model Accuracies
# ==========================================================
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Comparison - Customer Churn Prediction")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

best_model = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model} with Accuracy: {results[best_model]:.4f}")

# ==========================================================
# 5. Predict on New Customer Example
# ==========================================================
example = pd.DataFrame([{
    "CreditScore": 650,
    "Gender": 1,  # Male
    "Age": 35,
    "Tenure": 5,
    "Balance": 75000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 60000,
    "Geography_Germany": 0,
    "Geography_Spain": 1
}])

example_scaled = scaler.transform(example)
pred = models[best_model].predict(example_scaled)[0]
print("\nPrediction for new customer: ", "Churn" if pred == 1 else "Not Churn")
