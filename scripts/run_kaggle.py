import json
import os
import sys
import zipfile
import tempfile
from pathlib import Path

# ── Configure Kaggle auth before importing kaggle ─────────────────────────
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

with open(kaggle_json_path, "w") as f:
    json.dump({
        "username": os.environ["KAGGLE_USERNAME"],
        "key":      os.environ["KAGGLE_KEY"]
    }, f)
os.chmod(kaggle_json_path, 0o600)

# ── Now import kaggle ──────────────────────────────────────────────────────
from kaggle.api.kaggle_api_extended import KaggleApiExtended

api = KaggleApiExtended()
api.authenticate()

KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]
KAGGLE_DATASET  = "mlsecops-crack-seg-code"
KAGGLE_NOTEBOOK = "mlsecops-crack-seg-training"


def zip_repo(output_path: str):
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
    print("Pushing code to Kaggle dataset...")
    git_sha = os.environ.get("GIT_SHA", "unknown")

    meta_dir = tempfile.mkdtemp()
    metadata = {
        "title":     "mlsecops-crack-seg-code",
        "id":        f"{KAGGLE_USERNAME}/{KAGGLE_DATASET}",
        "licenses":  [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    with open(os.path.join(meta_dir, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(meta_dir)

    try:
        api.dataset_create_new(
            folder=meta_dir,
            public=False,
            quiet=False,
            convert_to_csv=False,
            dir_mode="zip"
        )
        print("Dataset created successfully.")
    except Exception as e:
        if "already exists" in str(e).lower() or "403" in str(e):
            api.dataset_create_version(
                folder=meta_dir,
                version_notes=f"CT update - {git_sha[:7]}",
                quiet=False,
                convert_to_csv=False,
                delete_old_versions=False,
                dir_mode="zip"
            )
            print("Dataset version updated successfully.")
        else:
            print(f"ERROR pushing dataset: {e}")
            sys.exit(1)


def trigger_notebook():
    print("Triggering Kaggle notebook...")

    git_sha    = os.environ.get("GIT_SHA", "unknown")
    aws_key_id = os.environ.get("CT_AWS_ACCESS_KEY_ID", "")
    aws_secret = os.environ.get("CT_AWS_SECRET_ACCESS_KEY", "")

    notebook_source = open("scripts/kaggle_notebook.ipynb").read()

    kernel_meta_dir = tempfile.mkdtemp()
    kernel_metadata = {
        "id":                  f"{KAGGLE_USERNAME}/{KAGGLE_NOTEBOOK}",
        "title":               KAGGLE_NOTEBOOK,
        "code_file":           "kaggle_notebook.ipynb",
        "language":            "python",
        "kernel_type":         "notebook",
        "is_private":          True,
        "enable_gpu":          True,
        "enable_internet":     True,
        "dataset_sources":     [f"{KAGGLE_USERNAME}/{KAGGLE_DATASET}"],
        "competition_sources": [],
        "kernel_sources":      [],
        "environment_variables": [
            {"key": "GIT_SHA",                  "value": git_sha},
            {"key": "CT_AWS_ACCESS_KEY_ID",     "value": aws_key_id},
            {"key": "CT_AWS_SECRET_ACCESS_KEY", "value": aws_secret},
        ]
    }

    with open(os.path.join(kernel_meta_dir, "kernel-metadata.json"), "w") as f:
        json.dump(kernel_metadata, f)

    with open(os.path.join(kernel_meta_dir, "kaggle_notebook.ipynb"), "w") as f:
        f.write(notebook_source)

    api.kernels_push(kernel_meta_dir)
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