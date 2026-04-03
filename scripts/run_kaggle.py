"""
run_kaggle.py — Called by GitHub Actions to:
1. Push repo code as a Kaggle dataset
2. Trigger the training notebook
"""

import os
import sys
import json
import zipfile
import tempfile
import requests
from pathlib import Path

# ── Kaggle API credentials ─────────────────────────────────────────────────
KAGGLE_TOKEN    = os.environ["KAGGLE_API_TOKEN"]
KAGGLE_USERNAME = "nasriabidi"
KAGGLE_DATASET  = "mlsecops-crack-seg-code"
KAGGLE_NOTEBOOK = "mlsecops-crack-seg-training"

# New Kaggle API token format uses this header
HEADERS = {
    "Content-Type": "application/json",
    "X-Kaggle-Key": KAGGLE_TOKEN
}

API = "https://www.kaggle.com/api/v1"


def get_session():
    """Create requests session with correct Kaggle auth."""
    session = requests.Session()
    session.headers.update({"X-Kaggle-Key": KAGGLE_TOKEN})
    return session


def zip_repo(output_path: str):
    """Zip all relevant repo files to push to Kaggle as a dataset."""
    files_to_include = [
        "train.py",
        "validate_dataset.py",
        "crack-seg.yaml",
        "crack-seg.dvc",
        "requirements.txt",
        ".dvc/config",
        ".dvcignore",
    ]
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_include:
            path = Path(f)
            if path.exists():
                zf.write(path, path)
                print(f"  Added: {f}")
            else:
                print(f"  WARNING: {f} not found, skipping")
    print(f"Zip created: {output_path}")


def push_dataset(zip_path: str):
    """Push zipped code as a Kaggle dataset (creates or updates)."""
    print("Pushing code to Kaggle dataset...")
    session = get_session()

    git_sha = os.environ.get("GIT_SHA", "unknown")

    # Check if dataset exists
    r = session.get(f"{API}/datasets/{KAGGLE_USERNAME}/{KAGGLE_DATASET}")
    print(f"Dataset check status: {r.status_code}")

    with open(zip_path, "rb") as f:
        files = {"file": (os.path.basename(zip_path), f, "application/zip")}

        if r.status_code == 200:
            # Update existing dataset — new version
            r2 = session.post(
                f"{API}/datasets/{KAGGLE_USERNAME}/{KAGGLE_DATASET}/versions",
                data={"versionNotes": f"CT update - {git_sha[:7]}"},
                files=files
            )
        else:
            # Create new dataset
            metadata = {
                "title":    "mlsecops-crack-seg-code",
                "id":       f"{KAGGLE_USERNAME}/{KAGGLE_DATASET}",
                "licenses": [{"name": "CC0-1.0"}],
                "isPrivate": True,
            }
            r2 = session.post(
                f"{API}/datasets",
                data={"body": json.dumps(metadata)},
                files=files
            )

    print(f"Dataset push status: {r2.status_code}")
    if r2.status_code not in (200, 201):
        print(f"ERROR: {r2.text[:500]}")
        sys.exit(1)
    print("Dataset pushed successfully.")


def trigger_notebook():
    """Push and run the training notebook on Kaggle T4 GPU."""
    print("Triggering Kaggle notebook...")
    session = get_session()

    git_sha    = os.environ.get("GIT_SHA", "unknown")
    aws_key_id = os.environ.get("CT_AWS_ACCESS_KEY_ID", "")
    aws_secret = os.environ.get("CT_AWS_SECRET_ACCESS_KEY", "")

    notebook_source = open("scripts/kaggle_notebook.ipynb").read()

    payload = {
        "title":          KAGGLE_NOTEBOOK,
        "text":           notebook_source,
        "language":       "python",
        "kernelType":     "notebook",
        "isPrivate":      True,
        "enableGpu":      True,
        "enableInternet": True,
        "datasetDataSources": [
            f"{KAGGLE_USERNAME}/{KAGGLE_DATASET}"
        ],
        "environmentVariables": [
            {"key": "GIT_SHA",                  "value": git_sha},
            {"key": "CT_AWS_ACCESS_KEY_ID",     "value": aws_key_id},
            {"key": "CT_AWS_SECRET_ACCESS_KEY", "value": aws_secret},
        ]
    }

    r = session.post(f"{API}/kernels/push", json=payload)
    print(f"Notebook trigger status: {r.status_code}")

    if r.status_code not in (200, 201):
        print(f"ERROR: {r.text[:500]}")
        sys.exit(1)

    print("Notebook triggered successfully.")


def main():
    git_sha = os.environ.get("GIT_SHA", "unknown")
    print(f"Git SHA: {git_sha}")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name

    zip_repo(zip_path)
    push_dataset(zip_path)
    trigger_notebook()

    print("Done. GitHub Actions will now poll S3 for the model.")


if __name__ == "__main__":
    main()