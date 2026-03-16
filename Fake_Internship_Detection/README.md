# 🔍 Fake Internship Detection Using Machine Learning

> **A College Data Science Mini Project**  
> Detects whether an internship/job posting is **Fake** or **Legitimate** using NLP and ML.

---

## 📋 Problem Statement

Every day, thousands of students and job seekers fall victim to fake internship and job postings online. Scammers craft convincing-looking posts to steal personal data, demand registration fees, or commit financial fraud. This project uses **Natural Language Processing (NLP)** and **Machine Learning** to automatically classify a job posting as **Fake** or **Legitimate**, helping protect users before they apply.

---

## 🎯 Objectives

- Detect fake internship/job postings automatically
- Analyse text features in job descriptions using NLP
- Apply TF-IDF vectorization to convert text to numbers
- Train and compare multiple ML classifiers
- Build a user-friendly prediction interface

---

## 📦 Dataset

- **Source:** [Kaggle – Real or Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobpostings)
- **File:** `dataset/fake_job_postings.csv`
- **Size:** ~17,880 job postings
- **Target:** `fraudulent` column (0 = Legitimate, 1 = Fake)

### Key Columns

| Column | Description |
|---|---|
| `title` | Job/internship title |
| `company_profile` | About the company |
| `description` | Full job description |
| `requirements` | Skills and qualifications required |
| `location` | Job location |
| `salary_range` | Salary info (often missing in fakes) |
| `fraudulent` | **Target: 1 = Fake, 0 = Real** |

---

## 🏗️ Project Architecture

```
Dataset (CSV)
    │
    ▼
Data Cleaning (pandas)
    │  • Remove nulls & duplicates
    │  • Select key columns
    ▼
Feature Engineering
    │  • Combine title + description + requirements
    ▼
NLP Text Processing (NLTK)
    │  • Lowercase, remove punctuation/stopwords
    │  • Tokenize
    ▼
TF-IDF Vectorization (sklearn)
    │  • Convert text → numerical matrix
    ▼
ML Model Training
    │  • Logistic Regression
    │  • Naive Bayes
    │  • Random Forest
    │  • SVM
    ▼
Model Evaluation
    │  • Accuracy, Precision, Recall, F1
    │  • Confusion Matrix
    ▼
Prediction System
    │  • Input description → Fake/Legitimate
    ▼
Streamlit Web App (Optional)
```

---

## 📂 Folder Structure

```
Fake_Internship_Detection/
│
├── dataset/
│   └── fake_job_postings.csv        ← Download from Kaggle
│
├── notebook/
│   └── internship_detection.ipynb   ← Main Jupyter Notebook
│
├── models/
│   └── trained_model.pkl            ← Saved best model
│   └── tfidf_vectorizer.pkl         ← Saved TF-IDF vectorizer
│
├── app/
│   ├── prediction_script.py         ← Command-line prediction tool
│   └── streamlit_app.py             ← Web app interface
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
Download `fake_job_postings.csv` from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobpostings) and place it in the `dataset/` folder.

### Step 3: Run the Notebook
Open `notebook/internship_detection.ipynb` in Jupyter Notebook or Google Colab and run all cells.

### Step 4: Run Prediction Script
```bash
python app/prediction_script.py
```

### Step 5: Run Web App (Optional)
```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 Models Used

| Model | Description |
|---|---|
| Logistic Regression | Fast linear classifier, great baseline |
| Naive Bayes | Probabilistic, excellent for text data |
| Random Forest | Ensemble of decision trees, robust |
| SVM | Finds optimal separating hyperplane |

---

## 📊 Sample Results

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | ~97% | ~0.96 |
| Naive Bayes | ~96% | ~0.95 |
| Random Forest | ~98% | ~0.97 |
| SVM | ~97% | ~0.96 |

*(Actual results may vary depending on dataset version)*

---

## 👨‍💻 Technologies Used

- **Python 3.x**
- **Pandas & NumPy** – Data manipulation
- **NLTK** – Natural Language Processing
- **Scikit-learn** – Machine Learning models & TF-IDF
- **Matplotlib & Seaborn** – Visualizations
- **Streamlit** – Web interface

---

## 📝 Conclusion

This project demonstrates how NLP and Machine Learning can effectively identify fraudulent job postings, achieving up to **98% accuracy**. The system can be used as a browser extension, API, or web app to protect job seekers in real time.

---

*Made with ❤️ as a Data Science Mini Project*
