"""
train_and_evaluate.py
Full ML training pipeline for the Fake Internship Detection System.
Trains 4 models, evaluates them, saves plots, and saves the best model.
Run: python train_and_evaluate.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
import os
import joblib

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# STEP 1: Load Dataset
# =============================================================================
print("=" * 55)
print(" FAKE INTERNSHIP DETECTION — TRAINING PIPELINE")
print("=" * 55)

df = pd.read_csv(os.path.join(BASE, "dataset", "fake_job_postings.csv"))
print(f"\n[1/8] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"      Legitimate: {(df['fraudulent']==0).sum()}")
print(f"      Fake      : {(df['fraudulent']==1).sum()}")

# =============================================================================
# STEP 2: Preprocessing
# =============================================================================
text_cols = ['title', 'company_profile', 'description', 'requirements',
             'location', 'salary_range']
for c in text_cols:
    if c in df.columns:
        df[c] = df[c].fillna('')

df['fraudulent'] = df['fraudulent'].astype(int)
before = len(df)
df = df.drop_duplicates()
print(f"\n[2/8] Preprocessing done. Removed {before-len(df)} duplicates.")

# =============================================================================
# STEP 3: Feature Engineering
# =============================================================================
df['text_data'] = (
    df['title'] + ' ' +
    df.get('company_profile', pd.Series([''] * len(df))).fillna('') + ' ' +
    df['description'] + ' ' +
    df['requirements'] + ' ' +
    df['location'] + ' ' +
    df['salary_range']
)
print(f"[3/8] Feature engineering done. Combined text_data column created.")

# =============================================================================
# STEP 4: NLP Text Cleaning
# =============================================================================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

print(f"[4/8] Cleaning text... ", end='', flush=True)
df['clean_text'] = df['text_data'].apply(clean_text)
print(f"Done. Avg token length: {df['clean_text'].str.split().str.len().mean():.0f} words/doc")

# =============================================================================
# STEP 5: TF-IDF + Train-Test Split
# =============================================================================
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X = tfidf.fit_transform(df['clean_text'])
y = df['fraudulent']
print(f"[5/8] TF-IDF vectorized: matrix shape = {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"      Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# =============================================================================
# STEP 6: Train Models
# =============================================================================
print(f"\n[6/8] Training models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Naive Bayes":         MultinomialNB(alpha=0.1),
    "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=20,
                                                   random_state=42, n_jobs=-1),
    "SVM (LinearSVC)":     LinearSVC(C=1.0, max_iter=2000, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    results[name] = dict(model=model, y_pred=y_pred,
                         accuracy=acc, precision=prec, recall=rec, f1_score=f1)
    print(f"      {name:25s}  Acc={acc*100:.1f}%  F1={f1*100:.1f}%")

# =============================================================================
# STEP 7: Save Charts
# =============================================================================
print(f"\n[7/8] Saving charts...")
models_dir = os.path.join(BASE, "models")
os.makedirs(models_dir, exist_ok=True)

# --- Distribution plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Fake vs Legitimate Job Postings', fontsize=14, fontweight='bold')
counts = df['fraudulent'].value_counts()
axes[0].bar(['Legitimate', 'Fake'], counts.values, color=['#2ecc71','#e74c3c'], edgecolor='black')
axes[0].set_title('Count')
axes[0].set_ylabel('Number of Postings')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')
axes[1].pie(counts.values, labels=['Legitimate','Fake'],
            colors=['#2ecc71','#e74c3c'], autopct='%1.1f%%', startangle=140)
axes[1].set_title('Proportion')
plt.tight_layout()
plt.savefig(os.path.join(models_dir, 'distribution_plot.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Model accuracy comparison ---
model_names = list(results.keys())
cols = ['#3498db','#2ecc71','#e74c3c','#f39c12']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

x = np.arange(len(model_names)); w = 0.2
for i, (metric, key) in enumerate([('Accuracy','accuracy'),('Precision','precision'),
                                     ('Recall','recall'),('F1','f1_score')]):
    vals = [results[m][key] for m in model_names]
    axes[0].bar(x + i*w, vals, w, label=metric, color=cols[i], alpha=0.85)
axes[0].set_xticks(x + w*1.5)
axes[0].set_xticklabels(model_names, rotation=15, ha='right', fontsize=8)
axes[0].set_ylim(0.7, 1.05); axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=0.3)
axes[0].set_title('All Metrics')

accs = [results[m]['accuracy']*100 for m in model_names]
bars = axes[1].barh(model_names, accs,
                    color=['#8e44ad','#16a085','#c0392b','#d35400'], edgecolor='black', height=0.5)
axes[1].set_xlim(50, 105); axes[1].set_title('Accuracy (%)')
for bar, val in zip(bars, accs):
    axes[1].text(val+0.5, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(models_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Confusion matrices ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
for idx, (name, res) in enumerate(results.items()):
    ax  = axes[idx//2][idx%2]
    cm  = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred Real','Pred Fake'],
                yticklabels=['True Real','True Fake'])
    ax.set_title(f'{name}  (Acc={res["accuracy"]*100:.1f}%)', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(models_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("      Charts saved to models/")

# =============================================================================
# STEP 8: Save Best Model
# =============================================================================
best_name  = max(results, key=lambda m: results[m]['f1_score'])
best_model = results[best_name]['model']

joblib.dump(best_model, os.path.join(models_dir, 'trained_model.pkl'))
joblib.dump(tfidf,      os.path.join(models_dir, 'tfidf_vectorizer.pkl'))

print(f"\n[8/8] Best model saved: {best_name}")
print(f"      F1 Score : {results[best_name]['f1_score']*100:.1f}%")
print(f"      Accuracy : {results[best_name]['accuracy']*100:.1f}%")

# =============================================================================
# CLASSIFICATION REPORT
# =============================================================================
print(f"\n{'='*55}")
print(f" CLASSIFICATION REPORT — {best_name}")
print(f"{'='*55}")
print(classification_report(y_test, results[best_name]['y_pred'],
                            target_names=['Legitimate', 'Fake']))

# =============================================================================
# QUICK PREDICTION TEST
# =============================================================================
def predict(text, model=best_model, vectorizer=tfidf):
    cleaned    = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred       = model.predict(vectorized)[0]
    return "FAKE — DO NOT APPLY" if pred == 1 else "LEGITIMATE — Looks Safe"

print(f"\n{'='*55}")
print(" QUICK PREDICTION TESTS")
print(f"{'='*55}")
tests = [
    ("FAKE TEST",  "Work from home. Pay Rs.999 registration fee. Earn Rs.50,000 guaranteed. No experience needed."),
    ("LEGIT TEST", "Software Engineering Intern at TechCorp. Python skills required. Stipend Rs.15,000/month. 3-month internship."),
]
for label, text in tests:
    print(f"  {label}: {text[:60]}...")
    print(f"  Result => {predict(text)}\n")

print("Training pipeline complete! Models saved in models/")
print("Next: run  streamlit run app/streamlit_app.py")
