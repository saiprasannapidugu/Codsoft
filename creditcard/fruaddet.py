import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve
)
import seaborn as sns
import tkinter as tk
from tkinter import messagebox

# ---------- Locate dataset ----------
possible_paths = [
    "fraudTrain.csv",
    "fraudTest.csv",
    os.path.expanduser("~/Downloads/archive/fraudTrain.csv"),
    os.path.expanduser("~/Downloads/archive/fraudTest.csv"),
    os.path.expanduser("~/Downloads/fraudTrain.csv"),
    os.path.expanduser("~/Downloads/fraudTest.csv"),
    r"C:\Users\Admin\OneDrive\Desktop\python\fraudTrain.csv",
    r"C:\Users\Admin\OneDrive\Desktop\python\fraudTest.csv"
]

train_path, test_path = None, None
for path in possible_paths:
    if os.path.exists(path) and "Train" in os.path.basename(path):
        train_path = path
    if os.path.exists(path) and "Test" in os.path.basename(path):
        test_path = path

if not train_path or not test_path:
    raise FileNotFoundError("Could not find fraudTrain.csv and fraudTest.csv. "
                            "Please place them in the same folder as this script or update the paths.")

# ---------- Load dataset ----------
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df = pd.concat([df_train, df_test], ignore_index=True)

# ---------- Identify label column ----------
label_col = None
for c in ['Class', 'class', 'is_fraud', 'isFraud', 'fraud', 'label', 'target']:
    if c in df.columns:
        label_col = c
        break

if label_col and label_col != 'Class':
    df = df.rename(columns={label_col: 'Class'})

# ---------- Clean and preprocess ----------
drop_cols = [
    'trans_date_trans_time', 'unix_time', 'merchant', 'first', 'last',
    'street', 'city', 'state', 'cc_num', 'dob', 'job', 'trans_num'
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

X = df.select_dtypes(include=[np.number]).drop(columns=['Class'], errors='ignore')
y = df['Class']

# Subsample for balance
pos_idx = y[y == 1].index
neg_idx = y[y == 0].index
np.random.seed(42)
neg_sample = np.random.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 3), replace=False)
sample_idx = np.concatenate([pos_idx, neg_sample])
X = X.loc[sample_idx].reset_index(drop=True)
y = y.loc[sample_idx].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------- Evaluation Function ----------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_scores)
    }

    p, r, _ = precision_recall_curve(y_test, y_scores)
    metrics['pr_auc'] = auc(r, p)
    metrics['fpr'], metrics['tpr'], _ = roc_curve(y_test, y_scores)
    metrics['precision_curve'], metrics['recall_curve'] = p, r

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{name}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, "
          f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, "
          f"PR-AUC={metrics['pr_auc']:.4f}")
    print("Confusion Matrix:\n", cm)

    return metrics

# ---------- Train Models ----------
results = {}
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='saga', max_iter=2000, class_weight='balanced', random_state=42))
])
lr.fit(X_train, y_train)
results['Logistic Regression'] = evaluate_model("Logistic Regression", lr, X_test, y_test)

dt = DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=8)
dt.fit(X_train, y_train)
results['Decision Tree'] = evaluate_model("Decision Tree", dt, X_test, y_test)

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced', max_depth=10)
rf.fit(X_train, y_train)
results['Random Forest'] = evaluate_model("Random Forest", rf, X_test, y_test)

# ---------- Model Comparison ----------
metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
comparison_df = pd.DataFrame([
    {'Model': name, **{m: results[name][m] for m in metrics_to_compare}}
    for name in results
])

# ---------- Plotting in separate popup window ----------
plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics_to_compare):
    plt.bar([x + i * 0.12 for x in range(len(results))],
            [results[m][metric] for m in results],
            width=0.12, label=metric.capitalize())
plt.xticks([x + 0.3 for x in range(len(results))], list(results.keys()))
plt.title("Model Comparison Across Metrics")
plt.legend()
plt.tight_layout()

# ROC Curve
plt.figure(figsize=(6, 5))
for name, res in results.items():
    plt.plot(res['fpr'], res['tpr'], label=f"{name} (AUC={res['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curves")
plt.legend()

# Precision-Recall Curve
plt.figure(figsize=(6, 5))
for name, res in results.items():
    plt.plot(res['recall_curve'], res['precision_curve'], label=f"{name} (AUC={res['pr_auc']:.3f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Precision-Recall Curves")
plt.legend()

# ---------- Tkinter Popup ----------
root = tk.Tk()
root.withdraw()  # Hide main window

msg = "Model Performance Summary:\n\n" + comparison_df.to_string(index=False)
messagebox.showinfo("Fraud Detection Results", msg)

plt.show()
