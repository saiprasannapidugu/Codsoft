# Codsoft
Task 1 — Movie Genre Classification

Goal: Predict one or multiple movie genres from a plot summary.

Dataset: DATASET CLICK HERE (replace with actual dataset URL). Example datasets: [IMDb plot summaries / MovieLens augmented with plots / Kaggle movie plot datasets].
Notes: If the dataset has multi-label genres (movies can belong to multiple genres) the project supports multi-label classification.

Preprocessing

Clean text (lowercase, remove punctuation).

Tokenize; remove stopwords (optional).

Use TF-IDF vectors or pre-trained word embeddings (e.g., Word2Vec / GloVe) and average/sequence encoding.

Optionally try text length features, presence of named entities, etc.

Models implemented

Multinomial Naive Bayes (TF-IDF)

Logistic Regression (TF-IDF / embeddings)

SVM (TF-IDF)

Optional: simple neural network (Keras) or fine-tune a transformer (if data & GPU available)

How to train

python src/task1/train.py --data data/movie_plots.csv --model_type lr --output models/task1_lr.pkl


Evaluation

Accuracy, Precision, Recall, F1-score (macro/micro); for multi-label tasks use micro/macro F1 and Hamming loss.

Confusion matrix (for single-label) and classification report exported to reports/.

Example inference

python src/task1/infer.py --model models/task1_lr.pkl --text "Two childhood friends reunite for a last adventure..."

Task 2 — Credit Card Fraud Detection

Goal: Classify transactions as fraudulent or legitimate.

Dataset: DATASET CLICK HERE (replace with actual dataset URL). Common example: the Kaggle credit card fraud dataset (anonymized features).
Notes: The dataset is typically highly imbalanced — address imbalance carefully.

Preprocessing

Handle class imbalance: undersampling, oversampling (SMOTE), class weights.

Feature scaling (StandardScaler) if required for models.

Time-window features, transaction amount transformations, aggregated user-level features (if available).

Models implemented

Logistic Regression (with class_weight)

Decision Tree / Random Forest

XGBoost

Baseline: simple thresholding on amount/time for sanity check

How to train

python src/task2/train.py --data data/creditcard.csv --model_type xgb --output models/task2_xgb.pkl


Evaluation

Use precision, recall, F1 for fraud class (positive), and especially Precision-Recall AUC and ROC-AUC.

Confusion matrix and classification report saved to reports/metrics/task2_*.json and plots in reports/.

Important tips

Prefer recall (catch frauds) while maintaining reasonable precision — business needs trade-off.

Use stratified splits and cross-validation that preserves class imbalance.

Example inference

python src/task2/infer.py --model models/task2_xgb.pkl --sample "path/to/sample.json"

Task 3 — Customer Churn Prediction

Goal: Predict which customers will churn (cancel subscription) in a future period.

Dataset: DATASET CLICK HERE (replace with the dataset link). Typical fields: customer_id, tenure, monthly_charges, total_charges, usage_metrics, demographic features, churn_label.

Preprocessing

Handle missing values, encode categorical variables (One-Hot, Target Encoding).

Feature engineering: tenure buckets, usage trends, contract type, payment method.

Normalize numeric features for algorithms that need it.

Models implemented

Logistic Regression

Random Forest

Gradient Boosting (XGBoost / LightGBM)

How to train

python src/task3/train.py --data data/churn.csv --model_type lgbm --output models/task3_lgbm.pkl


Evaluation

Precision, Recall, F1, ROC-AUC.

Business-oriented metrics: lift, gain, confusion matrix at chosen threshold, expected retention cost savings.

Example inference

python src/task3/infer.py --model models/task3_lgbm.pkl --customer "12345"

Evaluation metrics & model selection

Use stratified train/test splits and cross-validation.

For imbalanced tasks (fraud, churn): focus on Precision-Recall metrics and class-specific F1.

Save model checkpoints and scaler/vectorizer objects (use joblib or pickle), and include a small predict.py or infer.py for production-style inference.

Files to look for / example outputs

notebooks/ — exploratory analysis and sample training runs with plots.

src/*/train.py — command-line training scripts (accept --data, --model_type, --output).

src/*/infer.py — simple inference script to test single examples.

models/ — saved models (not committed to git for large models; add to .gitignore).

reports/ — saved metrics and plots (example confusion matrices, ROC/PR curves).

Reproducibility & tips

Set a random seed in each script (numpy, random, sklearn) for reproducibility.

Log hyperparameters and results to a CSV or a simple experiment tracker.

Save vectorizers/tokenizers and scalers alongside models so inference pipeline matches training pipeline.

How to contribute

Fork the repository.

Create a branch: feature/your-feature.

Add code/tests/docs.

Open a Pull Request with a clear description of changes.

Please follow code style and include brief docstrings for new functions.

License & citation

This repository is provided under the MIT License (or choose whichever license you prefer). Add LICENSE file with full text.

If you use the code in academic or production work, please cite this repo and include links to the original dataset(s) used.
