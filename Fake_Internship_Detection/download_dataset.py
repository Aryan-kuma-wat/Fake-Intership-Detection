"""
download_dataset.py  —  Downloads the real Kaggle dataset interactively.

HOW TO GET YOUR KAGGLE API KEY:
  1. Go to https://www.kaggle.com  (create a free account if needed)
  2. Click your profile picture → "Settings"
  3. Scroll to "API" section → click "Create New Token"
  4. A file called kaggle.json is downloaded — keep it handy
  5. Run: python download_dataset.py
  6. Enter your Kaggle username and the API key from kaggle.json when prompted

The script will download fake_job_postings.csv into the dataset/ folder.
"""

import os
import sys
import shutil

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

print("=" * 60)
print("  Kaggle Dataset Downloader")
print("  Dataset: Real or Fake Job Postings (shivamb/real-or-fake-fake-jobpostings)")
print("=" * 60)
print()

target_csv = os.path.join(DEST, "fake_job_postings.csv")
if os.path.exists(target_csv):
    size = os.path.getsize(target_csv) / 1024
    if size > 500:   # real dataset is ~1.8 MB, synthetic is ~200 KB
        print(f"Dataset already exists ({size/1024:.1f} MB). Looks like the real dataset!")
        print("If you want to re-download, delete dataset/fake_job_postings.csv first.")
        sys.exit(0)
    else:
        print(f"Found existing CSV ({size:.0f} KB) — this is the synthetic version.")
        print("Downloading the real Kaggle dataset now...\n")

try:
    import opendatasets as od
    print("Using opendatasets to download...")
    print("You will be prompted for your Kaggle username and API key.\n")
    od.download(
        "https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobpostings",
        data_dir=DEST
    )
    # opendatasets saves to a subfolder — move the CSV up
    sub = os.path.join(DEST, "real-or-fake-fake-jobpostings")
    if os.path.isdir(sub):
        for f in os.listdir(sub):
            if f.endswith(".csv"):
                shutil.move(os.path.join(sub, f), target_csv)
        shutil.rmtree(sub, ignore_errors=True)

    if os.path.exists(target_csv):
        size = os.path.getsize(target_csv) / (1024*1024)
        print(f"\nDataset downloaded successfully! ({size:.1f} MB)")
        print(f"Saved to: {target_csv}")
        print("\nNow run:  python train_and_evaluate.py")
    else:
        print("\nCSV not found after download. Check the dataset/ folder manually.")
except Exception as e:
    print(f"\nDownload failed: {e}")
    print()
    print("MANUAL STEPS:")
    print("  1. Go to: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobpostings")
    print("  2. Click 'Download' (zip file)")
    print("  3. Extract fake_job_postings.csv")
    print(f"  4. Copy it to: {DEST}")
    print("  5. Run: python train_and_evaluate.py")
