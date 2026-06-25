"""
Train SVM classifier for MBG sentiment analysis.

Pipeline:
  1. Load labelled CSV
  2. TF‑IDF vectorisation on cleaned text
  3. Stratified train / test split (80 / 20)
  4. SVC with linear kernel
  5. Evaluation (classification report + confusion matrix)
  6. Save model & vectoriser for later inference
"""
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ── Paths ──────────────────────────────────────────────────────────────────
LABELLED_CSV = "dataset/labeled/labeled.csv"
MODEL_DIR = "ml/models"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_sentiment.pkl")
VECTORISER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(LABELLED_CSV)
print(f"[LOAD] {len(df)} rows")

X = df["text"].astype(str)
y = df["label"]

print(f"[LABELS] {dict(y.value_counts())}")

# ── Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
print(f"[SPLIT] train={len(X_train)}  test={len(X_test)}")

# ── TF‑IDF vectorisation ───────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True,
)
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)
print(f"[TF-IDF] {X_train_tf.shape[1]} features")

# ── Hyperparameter tuning ──────────────────────────────────────────────────
from sklearn.model_selection import GridSearchCV

param_grid = {"C": [0.1, 1, 10, 100]}
grid = GridSearchCV(
    SVC(kernel="linear", class_weight="balanced", random_state=42),
    param_grid,
    cv=5,
    scoring="f1_macro",
    verbose=1,
)
grid.fit(X_train_tf, y_train)
print(f"[TUNING] Best params: {grid.best_params_}")
print(f"[TUNING] Best CV f1_macro: {grid.best_score_:.4f}")

model = grid.best_estimator_

# ── Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_tf)

print("\n" + "=" * 60)
print("Classification Report")
print("=" * 60)
print(classification_report(y_test, y_pred))

print("=" * 60)
print("Confusion Matrix")
print("=" * 60)
print(confusion_matrix(y_test, y_pred))
print()

# ── Save model & vectoriser ────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(tfidf, VECTORISER_PATH)
print(f"[SAVE] {MODEL_PATH}")
print(f"[SAVE] {VECTORISER_PATH}")

# ── Helper: predict new text ───────────────────────────────────────────────
def predict_sentiment(text: str) -> str:
    """Return sentiment label for a single Indonesian text."""
    vec = tfidf.transform([text])
    return model.predict(vec)[0]

print("\n[READY] predict_sentiment() available for inference")
print('        e.g. predict_sentiment("Program MBG sangat bermanfaat")')
