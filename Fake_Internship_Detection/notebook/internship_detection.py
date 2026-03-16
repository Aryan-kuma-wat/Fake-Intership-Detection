# =============================================================================
# FAKE INTERNSHIP DETECTION SYSTEM
# Complete Python Script - Compatible with Jupyter Notebook & Google Colab
# =============================================================================
# Run this script cell by cell in Jupyter Notebook or Google Colab
# Dataset: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobpostings
# =============================================================================


# ##############################################################################
# CELL 1: INSTALL & IMPORT LIBRARIES
# ##############################################################################

# Uncomment the line below if running in Google Colab
# !pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud joblib -q

import pandas as pd               # Data manipulation and analysis
import numpy as np                # Numerical computing
import matplotlib.pyplot as plt   # Plotting and visualization
import seaborn as sns             # Statistical data visualization
import re                         # Regular expressions for text cleaning
import string                     # String constants (punctuation etc.)
import warnings
import joblib                     # Save and load ML models

# NLTK for Natural Language Processing
import nltk
nltk.download('stopwords')        # Download English stopwords list
nltk.download('punkt')            # Download tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Scikit-learn for ML pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)

warnings.filterwarnings('ignore')  # Suppress unnecessary warnings

print("✅ All libraries imported successfully!")
print(f"   pandas   : {pd.__version__}")
print(f"   numpy    : {np.__version__}")
print(f"   sklearn  : __version__ check via import")


# ─── CELL 2: LOAD THE DATASET ─────────────────────────────────────────────────
# Auto-detects environment (Colab, Jupyter, or plain Python) and finds the CSV.

import os, sys

# Check if running in Google Colab
IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    pass

def find_dataset():
    """Try common locations for fake_job_postings.csv and return the first found path."""
    candidates = [
        'fake_job_postings.csv',             # Same folder as notebook, or Colab root
        '../dataset/fake_job_postings.csv',  # Jupyter inside notebook/ subfolder
        'dataset/fake_job_postings.csv',     # Project root
        '/content/fake_job_postings.csv',    # Colab explicit path
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

csv_path = find_dataset()

if csv_path is None:
    if IN_COLAB:
        print("Dataset not found. Please upload fake_job_postings.csv now:")
        print("Download from: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobpostings")
        from google.colab import files
        uploaded = files.upload()       # A file picker dialog will appear
        csv_path = list(uploaded.keys())[0]
        print(f"Uploaded: {csv_path}")
    else:
        raise FileNotFoundError(
            "\n\nDataset NOT found!\n"
            "Please download 'fake_job_postings.csv' from:\n"
            "  https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobpostings\n\n"
            "Then place it in ONE of these locations:\n"
            "  1. Fake_Internship_Detection/dataset/fake_job_postings.csv  (recommended)\n"
            "  2. Same folder as this notebook\n"
        )

df = pd.read_csv(csv_path)
print(f"Dataset loaded from : {csv_path}")
print(f"Shape               : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Fake postings       : {df['fraudulent'].sum()}")
print(f"Legitimate postings : {(df['fraudulent']==0).sum()}")
df.head()


# ##############################################################################
# CELL 3: EXPLORE THE DATASET
# ##############################################################################

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

# Show column names and data types
print("\n📊 Column Info:")
print(df.info())

print("\n📊 Column Names:")
print(df.columns.tolist())

print("\n📊 Basic Statistics:")
print(df.describe())

print("\n🎯 Target Variable Distribution:")
print(df['fraudulent'].value_counts())
print(f"\n   Legitimate (0): {df['fraudulent'].value_counts()[0]} postings")
print(f"   Fake      (1): {df['fraudulent'].value_counts()[1]} postings")

print("\n🔍 Missing Values per Column:")
print(df.isnull().sum())


# ##############################################################################
# CELL 4: VISUALIZE — Fake vs Legitimate Distribution
# ##############################################################################

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fake vs Legitimate Internship/Job Postings', fontsize=16, fontweight='bold')

# --- Plot 1: Count Plot ---
counts = df['fraudulent'].value_counts()
labels = ['Legitimate (0)', 'Fake (1)']
colors = ['#2ecc71', '#e74c3c']

axes[0].bar(labels, counts.values, color=colors, edgecolor='black', width=0.5)
axes[0].set_title('Count of Job Postings', fontsize=13)
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold', fontsize=12)

# --- Plot 2: Pie Chart ---
axes[1].pie(
    counts.values,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 12}
)
axes[1].set_title('Proportion of Fake vs Legitimate', fontsize=13)

plt.tight_layout()
plt.savefig('../models/distribution_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Distribution chart saved.")


# ##############################################################################
# CELL 5: DATA PREPROCESSING
# ##############################################################################

print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Step 1: Select key columns only
# We use text-based columns most relevant to detecting fraud
useful_columns = ['title', 'company_profile', 'description', 'requirements',
                  'location', 'salary_range', 'fraudulent']
df = df[useful_columns]
print(f"✅ Selected {len(useful_columns)} useful columns.")

# Step 2: Fill missing values with empty strings (for text columns)
text_columns = ['title', 'company_profile', 'description', 'requirements',
                'location', 'salary_range']
df[text_columns] = df[text_columns].fillna('')
print("✅ Missing text values filled with empty strings.")

# Step 3: Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"✅ Removed {before - after} duplicate rows. Remaining: {after}")

# Step 4: Ensure target column is integer type
df['fraudulent'] = df['fraudulent'].astype(int)
print(f"✅ Target variable type: {df['fraudulent'].dtype}")

print(f"\n📊 Dataset shape after preprocessing: {df.shape}")


# ##############################################################################
# CELL 6: FEATURE ENGINEERING — Combine Text Columns
# ##############################################################################

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Combine important text columns into one 'text_data' column
# This gives the model all relevant textual context at once
df['text_data'] = (
    df['title'] + ' ' +
    df['company_profile'] + ' ' +
    df['description'] + ' ' +
    df['requirements'] + ' ' +
    df['location'] + ' ' +
    df['salary_range']
)

print("✅ Created combined 'text_data' column.")
print(f"\nSample combined text (row 0):\n{df['text_data'][0][:300]}...")


# ##############################################################################
# CELL 7: NLP TEXT PREPROCESSING FUNCTION
# ##############################################################################

print("=" * 60)
print("NLP TEXT PROCESSING")
print("=" * 60)

# Load English stopwords (common words like 'the', 'is', 'and' that add no value)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans raw text for NLP processing.

    Steps:
    1. Lowercase all characters
    2. Remove URLs (http links)
    3. Remove HTML tags (e.g., <br>, <p>)
    4. Remove punctuation and special characters
    5. Remove digits/numbers
    6. Remove stopwords
    7. Strip extra whitespace

    Args:
        text (str): Raw input text

    Returns:
        str: Cleaned text string
    """
    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Step 3: Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Step 4: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 5: Remove numbers/digits
    text = re.sub(r'\d+', '', text)

    # Step 6: Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    # Step 7: Rejoin and strip whitespace
    return ' '.join(tokens).strip()


# Apply the cleaning function to our combined text column
print("⏳ Cleaning text... (this may take 30-60 seconds)")
df['clean_text'] = df['text_data'].apply(clean_text)
print("✅ Text cleaning complete!")

print(f"\nOriginal text sample:\n{df['text_data'][0][:200]}")
print(f"\nCleaned text sample:\n{df['clean_text'][0][:200]}")


# ##############################################################################
# CELL 8: TF-IDF VECTORIZATION
# ##############################################################################

print("=" * 60)
print("TF-IDF VECTORIZATION")
print("=" * 60)

# TF-IDF (Term Frequency–Inverse Document Frequency):
# Converts text into a numerical matrix where each word gets a score
# based on how frequently it appears in one document vs. all documents.
# Common words (like 'the') get LOW scores; unique/important words get HIGH scores.

tfidf = TfidfVectorizer(
    max_features=10000,    # Use top 10,000 most important words (features)
    ngram_range=(1, 2),    # Use single words AND two-word phrases (bigrams)
    sublinear_tf=True      # Apply log normalization to term frequency
)

# Fit the vectorizer on the cleaned text and transform to matrix
X = tfidf.fit_transform(df['clean_text'])

# Target variable (0 = Legitimate, 1 = Fake)
y = df['fraudulent']

print(f"✅ TF-IDF vectorization complete!")
print(f"   Feature matrix shape: {X.shape}")
print(f"   Number of text features (vocabulary): {X.shape[1]}")
print(f"   Number of samples: {X.shape[0]}")


# ##############################################################################
# CELL 9: TRAIN-TEST SPLIT
# ##############################################################################

print("=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

# Split data: 80% training, 20% testing
# random_state=42 ensures reproducibility (same split every run)
# stratify=y ensures both splits have same class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,        # 20% goes to test set
    random_state=42,       # For reproducibility
    stratify=y             # Keep class ratio balanced in both splits
)

print(f"✅ Data split complete!")
print(f"   Training samples : {X_train.shape[0]} ({80}%)")
print(f"   Testing  samples : {X_test.shape[0]} ({20}%)")
print(f"\n   Fake jobs in training  : {y_train.sum()}")
print(f"   Real jobs in training  : {(y_train == 0).sum()}")


# ##############################################################################
# CELL 10: TRAIN MULTIPLE ML MODELS
# ##############################################################################

print("=" * 60)
print("TRAINING MACHINE LEARNING MODELS")
print("=" * 60)

# Dictionary of models to train and compare
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,         # Allow up to 1000 iterations to converge
        C=1.0,                 # Regularization strength
        solver='lbfgs',        # Optimizer for multiclass problems
        random_state=42
    ),
    "Naive Bayes": MultinomialNB(
        alpha=0.1              # Smoothing parameter (handles unseen words)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,      # 100 decision trees in the ensemble
        max_depth=20,          # Max depth of each tree
        random_state=42,
        n_jobs=-1              # Use all CPU cores for speed
    ),
    "SVM (LinearSVC)": LinearSVC(
        C=1.0,                 # Regularization parameter
        max_iter=2000,         # Max iterations for convergence
        random_state=42
    )
}

# Store results for each model
results = {}

# Train each model and evaluate on test set
for model_name, model in models.items():
    print(f"\n⏳ Training {model_name}...")

    # Fit (train) the model on training data
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)

    # Store results
    results[model_name] = {
        'model':     model,
        'y_pred':    y_pred,
        'accuracy':  acc,
        'precision': prec,
        'recall':    rec,
        'f1_score':  f1
    }

    print(f"   ✅ {model_name} trained!")
    print(f"      Accuracy : {acc*100:.2f}%")
    print(f"      Precision: {prec*100:.2f}%")
    print(f"      Recall   : {rec*100:.2f}%")
    print(f"      F1 Score : {f1*100:.2f}%")


# ##############################################################################
# CELL 11: MODEL EVALUATION — Classification Reports
# ##############################################################################

print("=" * 60)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 60)

for model_name, res in results.items():
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print('='*50)
    print(classification_report(
        y_test, res['y_pred'],
        target_names=['Legitimate (0)', 'Fake (1)']
    ))


# ##############################################################################
# CELL 12: VISUALIZATION — Model Accuracy Comparison
# ##############################################################################

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

model_names = list(results.keys())
metrics = {
    'Accuracy':  [results[m]['accuracy']  for m in model_names],
    'Precision': [results[m]['precision'] for m in model_names],
    'Recall':    [results[m]['recall']    for m in model_names],
    'F1 Score':  [results[m]['f1_score']  for m in model_names],
}

# --- Plot 1: Grouped Bar Chart for all metrics ---
x = np.arange(len(model_names))
width = 0.2
bar_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

for i, (metric, values) in enumerate(metrics.items()):
    axes[0].bar(x + i * width, values, width, label=metric, color=bar_colors[i], alpha=0.85)

axes[0].set_title('All Metrics by Model', fontsize=13)
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
axes[0].set_ylim(0.8, 1.02)
axes[0].set_ylabel('Score')
axes[0].legend(fontsize=9)
axes[0].grid(axis='y', alpha=0.3)

# --- Plot 2: Accuracy-only horizontal bar chart ---
accuracies = [results[m]['accuracy'] * 100 for m in model_names]
bar_h = axes[1].barh(model_names, accuracies,
                     color=['#8e44ad', '#16a085', '#c0392b', '#d35400'],
                     edgecolor='black', height=0.5)
axes[1].set_title('Accuracy Comparison (%)', fontsize=13)
axes[1].set_xlabel('Accuracy (%)')
axes[1].set_xlim(85, 100)

for bar, val in zip(bar_h, accuracies):
    axes[1].text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{val:.2f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../models/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Model comparison chart saved.")


# ##############################################################################
# CELL 13: VISUALIZATION — Confusion Matrices
# ##############################################################################

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')
axes = axes.flatten()  # Flatten 2D array for easy indexing

for idx, (model_name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])

    sns.heatmap(
        cm,
        annot=True,              # Show numbers in cells
        fmt='d',                 # Integer format
        cmap='Blues',            # Blue color scheme
        ax=axes[idx],
        xticklabels=['Predicted Real', 'Predicted Fake'],
        yticklabels=['Actual Real', 'Actual Fake']
    )
    axes[idx].set_title(f'{model_name}\nAccuracy: {res["accuracy"]*100:.2f}%', fontsize=11)
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('../models/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrices saved.")


# ##############################################################################
# CELL 14: SAVE THE BEST MODEL
# ##############################################################################

print("=" * 60)
print("SAVING THE BEST MODEL")
print("=" * 60)

# Find the best model based on F1 Score
best_model_name = max(results, key=lambda m: results[m]['f1_score'])
best_model      = results[best_model_name]['model']

print(f"🏆 Best Model: {best_model_name}")
print(f"   F1 Score : {results[best_model_name]['f1_score']*100:.2f}%")
print(f"   Accuracy : {results[best_model_name]['accuracy']*100:.2f}%")

# Save model and vectorizer to disk for later use
joblib.dump(best_model, '../models/trained_model.pkl')
joblib.dump(tfidf,      '../models/tfidf_vectorizer.pkl')

print("\n✅ Model saved to: ../models/trained_model.pkl")
print("✅ Vectorizer saved to: ../models/tfidf_vectorizer.pkl")


# ##############################################################################
# CELL 15: FINAL PREDICTION SYSTEM
# ##############################################################################

print("=" * 60)
print("FINAL PREDICTION SYSTEM")
print("=" * 60)

def predict_internship(description, model=best_model, vectorizer=tfidf):
    """
    Predicts whether an internship posting is Fake or Legitimate.

    Args:
        description (str): Raw internship/job description text
        model: Trained ML model (default: best model)
        vectorizer: Fitted TF-IDF vectorizer

    Returns:
        str: "🚨 FAKE Internship" or "✅ Legitimate Internship"
    """
    # Step 1: Clean the input text using our preprocessing function
    cleaned = clean_text(description)

    # Step 2: Transform cleaned text using the fitted TF-IDF vectorizer
    vectorized = vectorizer.transform([cleaned])

    # Step 3: Predict using the trained model
    prediction = model.predict(vectorized)[0]

    # Step 4: Return human-readable result
    if prediction == 1:
        return "🚨 FAKE Internship — DO NOT Apply!"
    else:
        return "✅ Legitimate Internship — Looks Safe!"


# ---- TEST CASES ----

test_cases = [
    {
        "label": "Test 1 (Likely FAKE)",
        "text": "Work from home internship. No experience needed. Pay Rs.500 registration fee to start immediately. Earn Rs.50,000 per month guaranteed!"
    },
    {
        "label": "Test 2 (Likely FAKE)",
        "text": "Urgently hiring! Get paid $5000 per week working from home. No skills required. Send your Aadhar card and bank details to confirm your spot."
    },
    {
        "label": "Test 3 (Likely LEGITIMATE)",
        "text": "Software Engineering Intern at Google. We are looking for undergraduate students with knowledge of Python, Java, or C++. The intern will work on real projects, attend team meetings, and be mentored by senior engineers. Duration: 3 months. Stipend: Competitive."
    },
    {
        "label": "Test 4 (Likely LEGITIMATE)",
        "text": "Marketing Intern at ABC Solutions Pvt. Ltd. Responsibilities: Social media management, content creation, market research. Requirements: Good communication skills, enrolled in BBA/MBA. Duration: 6 months. Paid internship."
    },
]

print("\n🔍 Running Prediction Tests:\n")
for case in test_cases:
    result = predict_internship(case['text'])
    print(f"📌 {case['label']}")
    print(f"   Input : {case['text'][:80]}...")
    print(f"   Result: {result}")
    print()


# ##############################################################################
# CELL 16: INTERACTIVE PREDICTION (Run in Colab/Jupyter)
# ##############################################################################

print("=" * 60)
print("INTERACTIVE PREDICTION (enter your own description)")
print("=" * 60)

# Uncomment the lines below to interactively test your own descriptions
# user_input = input("🖊️  Paste the internship description here:\n> ")
# if user_input.strip():
#     result = predict_internship(user_input)
#     print(f"\n🎯 Prediction: {result}")
# else:
#     print("⚠️ No input provided.")


print("\n✅ All done! Project complete.")
print("📁 Check the 'models/' folder for saved plots and model files.")
