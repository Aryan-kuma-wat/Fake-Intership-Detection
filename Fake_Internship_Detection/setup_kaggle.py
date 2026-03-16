"""
setup_kaggle.py  —  One-step Kaggle setup + dataset download
Run: python setup_kaggle.py
"""
import os, json, sys

print("=" * 60)
print("  Kaggle API Setup for Fake Job Postings Dataset")
print("=" * 60)
print()
print("STEP 1: Get your Kaggle API key")
print("  1. Visit  https://www.kaggle.com  and log in (free account)")
print("  2. Click your profile photo → Settings")
print("  3. Scroll to 'API' section → 'Create New Token'")
print("  4. A file 'kaggle.json' downloads — open it and note the values")
print()

username = input("Enter your Kaggle username: ").strip()
api_key  = input("Enter your Kaggle API key  : ").strip()

if not username or not api_key:
    print("Credentials not entered. Exiting.")
    sys.exit(1)

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
cred_path = os.path.join(kaggle_dir, "kaggle.json")
with open(cred_path, "w") as f:
    json.dump({"username": username, "key": api_key}, f)
os.chmod(cred_path, 0o600)
print(f"\nCredentials saved to {cred_path}")

# Now download
import subprocess, shutil
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
os.makedirs(DEST, exist_ok=True)

print("\nDownloading dataset from Kaggle...")
result = subprocess.run(
    ["python", "-m", "kaggle", "datasets", "download",
     "-d", "shivamb/real-or-fake-fake-jobpostings",
     "-p", DEST, "--unzip"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("Error:", result.stderr)
    sys.exit(1)

csv_path = os.path.join(DEST, "fake_job_postings.csv")
if os.path.exists(csv_path):
    size = os.path.getsize(csv_path) / (1024*1024)
    print(f"Dataset downloaded! ({size:.1f} MB)")
    print("\nNow run:  python train_and_evaluate.py")
else:
    print("CSV not found after download. Check the dataset/ folder.")
