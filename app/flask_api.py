# =============================================================================
# FAKE INTERNSHIP DETECTION — FLASK API BACKEND
# =============================================================================
# Run: python app/flask_api.py
# Then open: http://127.0.0.1:5000
# =============================================================================

import os, re, string
from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'trained_model.pkl')
TFIDF_PATH = os.path.join(BASE_DIR, '..', 'models', 'tfidf_vectorizer.pkl')

# ── Flask app ─────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.join(BASE_DIR, '..')          # project root (one level up from app/)
app = Flask(__name__, static_folder='static')

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH) or not os.path.exists(TFIDF_PATH):
    raise FileNotFoundError("Model files not found. Run the Jupyter notebook first.")

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
stop_words = set(stopwords.words('english'))
print(f"[OK] Model loaded: {type(model).__name__}")

# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return ' '.join(w for w in text.split() if w not in stop_words and len(w) > 2)

# ── Red flag detection ────────────────────────────────────────────────────────
RED_FLAG_PATTERNS = {
    "Registration / security fee":    r'registrat|security fee|deposit|pay.*fee|fee.*pay',
    "Guaranteed salary / income":     r'guaranteed|assured.*(salary|income|payment)',
    "No experience required":         r'no experience|no qualification|anyone can',
    "Work from home (unverified)":    r'work from home|wfh|online part.?time',
    "Personal docs requested":        r'aadhar|pan card|bank detail|account number|passport',
    "Urgency / ASAP language":        r'urgent|asap|immediate|hurry|limited seat',
    "Unrealistic pay (>₹30k/month)":  r'(?:rs\.?\s*|₹\s*)([3-9]\d{4}|\d{5,})',
    "WhatsApp / phone contact":       r'whatsapp|call.*now|contact.*\+91|phone.*now',
}

def detect_red_flags(text):
    return {
        flag: bool(re.search(pattern, text, re.I))
        for flag, pattern in RED_FLAG_PATTERNS.items()
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    # Serve index.html from the project root (not inside app/templates)
    return send_file(os.path.join(ROOT_DIR, 'index.html'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    description = data.get('description', '').strip()

    if not description:
        return jsonify({'error': 'No description provided.'}), 400

    cleaned    = clean_text(description)
    vectorized = tfidf.transform([cleaned])
    prediction = int(model.predict(vectorized)[0])

    # Risk score (0–1)
    try:
        score     = model.decision_function(vectorized)[0]
        risk_score = float(1 / (1 + np.exp(-score)))   # sigmoid
    except AttributeError:
        try:
            risk_score = float(model.predict_proba(vectorized)[0][1])
        except AttributeError:
            risk_score = 0.85 if prediction == 1 else 0.15

    red_flags = detect_red_flags(description)
    flagged_count = sum(red_flags.values())

    return jsonify({
        'prediction':   prediction,
        'label':        'FAKE' if prediction == 1 else 'LEGITIMATE',
        'risk_score':   round(float(risk_score) * 100, 1),  # cast avoids type-checker warning
        'red_flags':    red_flags,
        'flagged_count': flagged_count,
        'word_count':   len(description.split()),
        'token_count':  len(cleaned.split()),
        'model_type':   type(model).__name__,
    })

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n[*] Starting FakeJob Shield server...")
    print("   Open: http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
