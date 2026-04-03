"""
run_kaggle.py — Called by GitHub Actions to:
1. Push repo code as a Kaggle dataset
2. Create/update the training notebook
3. Trigger notebook execution
4. Wait for completion (polls Kaggle API)
"""

import os
import sys
import time
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

HEADERS = {
    "Content-Type":  "application/json",
    "Authorization": f"Bearer {KAGGLE_TOKEN}"
}

API = "https://www.kaggle.com/api/v1"


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

    # Check if dataset exists
    r = requests.get(
        f"{API}/datasets/{KAGGLE_USERNAME}/{KAGGLE_DATASET}",
        headers=HEADERS
    )

    metadata = {
        "title":      KAGGLE_DATASET,
        "id":         f"{KAGGLE_USERNAME}/{KAGGLE_DATASET}",
        "licenses":   [{"name": "CC0-1.0"}],
        "isPrivate":  True,
    }

    with open(zip_path, "rb") as f:
        files = {"file": (zip_path, f, "application/zip")}
        headers_no_ct = {"Authorization": f"Bearer {KAGGLE_TOKEN}"}

        if r.status_code == 200:
            # Update existing dataset
            r2 = requests.post(
                f"{API}/datasets/{KAGGLE_USERNAME}/{KAGGLE_DATASET}/versions",
                headers=headers_no_ct,
                data={"versionNotes": f"CT update - {os.environ.get('GIT_SHA','unknown')}"},
                files=files
            )
        else:
            # Create new dataset
            r2 = requests.post(
                f"{API}/datasets",
                headers=headers_no_ct,
                data={"body": json.dumps(metadata)},
                files=files
            )

    if r2.status_code not in (200, 201):
        print(f"ERROR pushing dataset: {r2.status_code} {r2.text}")
        sys.exit(1)
    print("Dataset pushed successfully.")


def trigger_notebook():
    """Create or re-run the training notebook on Kaggle."""
    print("Triggering Kaggle notebook...")

    git_sha       = os.environ.get("GIT_SHA", "unknown")
    aws_key_id    = os.environ.get("CT_AWS_ACCESS_KEY_ID", "")
    aws_secret    = os.environ.get("CT_AWS_SECRET_ACCESS_KEY", "")

    notebook_payload = {
        "title":          KAGGLE_NOTEBOOK,
        "text":           open("scripts/kaggle_notebook.ipynb").read(),
        "language":       "python",
        "kernelType":     "notebook",
        "isPrivate":      True,
        "enableGpu":      True,
        "enableInternet": True,
        "datasetDataSources": [
            f"{KAGGLE_USERNAME}/{KAGGLE_DATASET}"
        ],
        "environmentVariables": [
            {"key": "GIT_SHA",                 "value": git_sha},
            {"key": "CT_AWS_ACCESS_KEY_ID",    "value": aws_key_id},
            {"key": "CT_AWS_SECRET_ACCESS_KEY", "value": aws_secret},
        ]
    }

    headers_no_ct = {"Authorization": f"Bearer {KAGGLE_TOKEN}"}

    # Check if notebook exists
    r = requests.get(
        f"{API}/kernels/{KAGGLE_USERNAME}/{KAGGLE_NOTEBOOK}",
        headers=HEADERS
    )

    if r.status_code == 200:
        # Push new version to re-run
        r2 = requests.post(
            f"{API}/kernels/push",
            headers=headers_no_ct,
            json=notebook_payload
        )
    else:
        # Create new notebook
        r2 = requests.post(
            f"{API}/kernels/push",
            headers=headers_no_ct,
            json=notebook_payload
        )

    if r2.status_code not in (200, 201):
        print(f"ERROR triggering notebook: {r2.status_code} {r2.text}")
        sys.exit(1)

    print("Notebook triggered successfully.")


def main():
    git_sha = os.environ.get("GIT_SHA", "unknown")
    print(f"Git SHA: {git_sha}")

    # Step 1 — Zip and push code to Kaggle
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name

    zip_repo(zip_path)
    push_dataset(zip_path)

    # Step 2 — Trigger notebook
    trigger_notebook()

    print("Kaggle notebook triggered. GitHub Actions will now poll S3 for model.")


if __name__ == "__main__":
    main()