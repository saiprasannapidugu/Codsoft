# ==========================================================
# Movie Genre Classification - TF-IDF + Logistic Regression
# Dataset: Genre_Classification_Dataset.csv
# Created directly from your uploaded dataset
# ==========================================================

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================================
# 1. Setup
# ==========================================================
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("Genre_Classification_Dataset.csv")
print(f"Dataset loaded successfully. Shape: {df.shape}")
print(df.head())

# ==========================================================
# 2. Preprocessing
# ==========================================================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

print("\nCleaning text data...")
df['clean_plot'] = df['Plot'].apply(clean_text)
print("✅ Text cleaning complete.\n")

# ==========================================================
# 3. Feature Extraction (TF-IDF)
# ==========================================================
print("Extracting TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_plot'])
y = df['Genre']

print(f"TF-IDF matrix shape: {X.shape}\n")

# ==========================================================
# 4. Train-Test Split
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training and test data split complete.\n")

# ==========================================================
# 5. Model Training
# ==========================================================
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)
print("✅ Model training complete.\n")

# ==========================================================
# 6. Evaluation
# ==========================================================
print("Evaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(12, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=False, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================================
# 7. Predict New Movie Genre
# ==========================================================
def predict_genre(plot_summary):
    cleaned = clean_text(plot_summary)
    vector = tfidf.transform([cleaned])
    return model.predict(vector)[0]

# Example
example_plot = "A young boy discovers he has magical powers and attends a school for wizards."
predicted_genre = predict_genre(example_plot)
print(f"\nExample prediction:\n{example_plot}\n➡ Predicted Genre: {predicted_genre}")
