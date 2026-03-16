# =============================================================================
# FAKE INTERNSHIP DETECTION — COMMAND-LINE PREDICTION SCRIPT
# =============================================================================
# Run this script AFTER running the main notebook (internship_detection.ipynb)
# The notebook MUST be run first to generate:
#   - models/trained_model.pkl
#   - models/tfidf_vectorizer.pkl
#
# Usage:
#   python prediction_script.py
# =============================================================================

import joblib          # For loading the saved ML model and vectorizer
import re              # Regular expressions for text cleaning
import string          # For removing punctuation
import nltk
import os

# Download NLTK data if not already downloaded
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# =============================================================================
# STEP 1: LOAD SAVED MODEL AND VECTORIZER
# =============================================================================

# Resolve paths relative to this script's location
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'trained_model.pkl')
TFIDF_PATH = os.path.join(BASE_DIR, '..', 'models', 'tfidf_vectorizer.pkl')

# Check if model files exist before loading
if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: trained_model.pkl not found!")
    print("   Please run the Jupyter Notebook first to train and save the model.")
    exit(1)

if not os.path.exists(TFIDF_PATH):
    print("❌ ERROR: tfidf_vectorizer.pkl not found!")
    print("   Please run the Jupyter Notebook first.")
    exit(1)

# Load the trained model from disk
model     = joblib.load(MODEL_PATH)
tfidf     = joblib.load(TFIDF_PATH)

print("✅ Model and vectorizer loaded successfully.")
print(f"   Model type: {type(model).__name__}")

# =============================================================================
# STEP 2: TEXT CLEANING FUNCTION
# =============================================================================

stop_words = set(stopwords.words('english'))  # English stopwords

def clean_text(text):
    """
    Cleans and preprocesses raw text input.

    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove HTML tags
        4. Remove punctuation
        5. Remove digits
        6. Remove stopwords
    """
    text = text.lower()                                           # Lowercase
    text = re.sub(r'http\S+|www\.\S+', '', text)                 # Remove URLs
    text = re.sub(r'<.*?>', '', text)                             # Remove HTML
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)                               # Remove digits
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# =============================================================================
# STEP 3: PREDICTION FUNCTION
# =============================================================================

def predict_internship(description):
    """
    Takes a raw internship description and returns a prediction.

    Args:
        description (str): Raw text of internship posting

    Returns:
        dict: Contains 'prediction' (int), 'label' (str), 'cleaned_text' (str)
    """
    if not description.strip():
        return {'error': 'Empty description provided.'}

    # Clean and vectorize the input text
    cleaned    = clean_text(description)
    vectorized = tfidf.transform([cleaned])

    # Make prediction (0 = Legitimate, 1 = Fake)
    prediction = model.predict(vectorized)[0]

    # Return result dictionary
    return {
        'prediction':   int(prediction),
        'label':        "🚨 FAKE Internship"      if prediction == 1 else "✅ Legitimate Internship",
        'advice':       "DO NOT apply or pay any fees!" if prediction == 1 else "Looks safe to apply!",
        'cleaned_text': cleaned[:200] + '...' if len(cleaned) > 200 else cleaned
    }

# =============================================================================
# STEP 4: MAIN INTERACTIVE LOOP
# =============================================================================

def main():
    """
    Main function: interactive command-line prediction loop.
    User can paste internship descriptions and get predictions.
    Type 'quit' or 'exit' to stop.
    """
    print("\n" + "=" * 62)
    print("  🔍 FAKE INTERNSHIP DETECTION SYSTEM")
    print("  Powered by Machine Learning + NLP")
    print("=" * 62)
    print("  Paste an internship/job description below.")
    print("  Type 'demo' to see example predictions.")
    print("  Type 'quit' or 'exit' to stop.\n")

    while True:
        print("-" * 62)
        user_input = input("📝 Enter internship description:\n> ").strip()

        # Exit condition
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("\n👋 Goodbye! Stay safe from fake internships!")
            break

        # Demo mode
        elif user_input.lower() == 'demo':
            demo_cases = [
                "Work from home. Pay Rs.500 registration fee. Earn Rs.50,000 guaranteed.",
                "Software Engineering Intern at Microsoft. Python/Java experience needed. Stipend provided.",
                "Urgently hiring! No skills required. Send bank details to get started immediately.",
                "Marketing Intern at ABC Corp. Content creation, social media management. Paid opportunity."
            ]
            print("\n📌 DEMO PREDICTIONS:\n")
            for i, text in enumerate(demo_cases, 1):
                res = predict_internship(text)
                print(f"  Demo {i}: {text[:60]}...")
                print(f"  Result: {res['label']} — {res['advice']}\n")

        # Empty input
        elif not user_input:
            print("⚠️  Please enter a description.")

        # Real prediction
        else:
            result = predict_internship(user_input)
            if 'error' in result:
                print(f"⚠️  Error: {result['error']}")
            else:
                print(f"\n{'='*40}")
                print(f"🎯 PREDICTION: {result['label']}")
                print(f"💡 ADVICE    : {result['advice']}")
                print(f"{'='*40}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    main()
