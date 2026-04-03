"""
run_kaggle.py — Triggers existing Kaggle notebook via API
No dataset upload needed — notebook clones repo directly from GitHub
"""

import json
import os
import sys
import tempfile

# ── Configure Kaggle auth ─────────────────────────────────────────────────
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
    json.dump({
        "username": os.environ["KAGGLE_USERNAME"],
        "key":      os.environ["KAGGLE_KEY"]
    }, f)
os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

# ── Import kaggle ─────────────────────────────────────────────────────────
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]
KAGGLE_NOTEBOOK = "mlsecops-crack-seg-training"


def trigger_notebook():
    print("Triggering Kaggle notebook...")

    git_sha    = os.environ.get("GIT_SHA", "unknown")
    aws_key_id = os.environ.get("CT_AWS_ACCESS_KEY_ID", "")
    aws_secret = os.environ.get("CT_AWS_SECRET_ACCESS_KEY", "")

    # Read the existing notebook source from Kaggle
    print("Pulling existing notebook from Kaggle...")
    kernel_meta_dir = tempfile.mkdtemp()

    # Pull existing kernel
    api.kernels_pull(
        kernel=f"{KAGGLE_USERNAME}/{KAGGLE_NOTEBOOK}",
        path=kernel_meta_dir,
        metadata=True
    )

    # Update kernel metadata with new env variables
    meta_path = os.path.join(kernel_meta_dir, "kernel-metadata.json")
    with open(meta_path, "r") as f:
        kernel_metadata = json.load(f)

    # Inject environment variables
    kernel_metadata["enable_gpu"]      = True
    kernel_metadata["enable_internet"] = True
    kernel_metadata["environment_variables"] = [
        {"key": "GIT_SHA",                  "value": git_sha},
        {"key": "CT_AWS_ACCESS_KEY_ID",     "value": aws_key_id},
        {"key": "CT_AWS_SECRET_ACCESS_KEY", "value": aws_secret},
    ]

    with open(meta_path, "w") as f:
        json.dump(kernel_metadata, f)

    # Push to trigger a new run
    api.kernels_push(kernel_meta_dir)
    print(f"Notebook triggered successfully.")
    print(f"GIT_SHA: {git_sha}")
    print(f"Track at: https://www.kaggle.com/code/{KAGGLE_USERNAME}/{KAGGLE_NOTEBOOK}")


def main():
    git_sha = os.environ.get("GIT_SHA", "unknown")
    print(f"Git SHA: {git_sha}")
    trigger_notebook()
    print("Done. GitHub Actions will now poll S3 for the model.")


if __name__ == "__main__":
    main()