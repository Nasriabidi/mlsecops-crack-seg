import sys
import os
import glob
import json
import subprocess
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import boto3
import mlflow
import whylogs as why
from ultralytics import YOLO

DATASET_PATH         = Path("./crack-seg")
DATASET_YAML         = "crack-seg.yaml"
MODEL_WEIGHTS        = "yolov8n-seg.pt"
MLFLOW_EXP_NAME      = "crack-seg-training"
MODELS_BUCKET        = os.environ.get("MODELS_BUCKET", "mlsecops-models-351611731527")
APP_REPO_URL         = os.environ.get("APP_REPO_URL", "")          # set in CT workflow
APP_REPO_TOKEN       = os.environ.get("APP_REPO_TOKEN", "")        # GitHub PAT secret

# ─────────────────────────────────────────────────────────────────────────────
# MLFLOW_TRACKING_URI must always be set — no silent SQLite fallback.
# ─────────────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    raise RuntimeError(
        "MLFLOW_TRACKING_URI environment variable is not set.\n"
        "Set it to your MLflow server URL, e.g.:\n"
        "  export MLFLOW_TRACKING_URI=http://<mlflow-server-ip>"
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_dvc_dataset_version() -> str:
    """Return current Git commit SHA — represents exact DVC dataset state."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log.warning("Could not get Git SHA for dataset_version, using 'unknown'")
        return "unknown"
    return result.stdout.strip()


def get_model_name(dataset_version: str) -> str:
    """Generate model filename: crack_seg_<sha7>_<YYYYMMDD>.pt"""
    sha_short = dataset_version[:7]
    timestamp = datetime.utcnow().strftime("%Y%m%d")
    return f"crack_seg_{sha_short}_{timestamp}.pt"


def pull_dataset():
    log.info("Pulling dataset from S3 via DVC...")
    result = subprocess.run(["dvc", "pull"], capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"DVC pull failed:\n{result.stderr}")
        sys.exit(1)
    log.info("Dataset pulled successfully.")


def validate_dataset():
    log.info("Running dataset validation...")
    result = subprocess.run(
        [sys.executable, "validate_dataset.py",
         "--dataset-path", str(DATASET_PATH),
         "--skip-resolution-check",
         "--ci"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        log.error("Dataset validation FAILED. Aborting training.")
        log.error(result.stderr)
        sys.exit(1)
    log.info("Dataset validation PASSED.")


# ── Training ──────────────────────────────────────────────────────────────────

def train(epochs: int, imgsz: int, batch: int, workers: int, model_name: str) -> tuple[Path, object]:
    log.info("=" * 55)
    log.info("  Starting YOLOv8n-seg Training")
    log.info(f"  epochs={epochs}  imgsz={imgsz}  batch={batch}  workers={workers}")
    log.info(f"  output model name: {model_name}")
    log.info("=" * 55)

    model = YOLO(MODEL_WEIGHTS)
    results = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        project="runs/segment",
        name="crack_seg",
        exist_ok=True,
    )

    matches = [
        m for m in glob.glob("runs/**/best.pt", recursive=True)
        if "mlflow" not in m
    ]
    if not matches:
        log.error("Training finished but best.pt not found anywhere in runs/")
        sys.exit(1)

    best_weights = Path(matches[0])
    log.info(f"Found best.pt at: {best_weights}")

    named_weights = best_weights.parent / model_name
    best_weights.rename(named_weights)
    log.info(f"Model renamed and saved as: {named_weights}")

    return named_weights, results


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(
    results,
    best_weights: Path,
    dataset_version: str,
    epochs: int,
    imgsz: int,
    batch: int,
) -> tuple[str, dict]:
    """Log to MLflow. Returns (run_id, metrics_dict)."""
    log.info(f"Logging experiment to MLflow server: {MLFLOW_TRACKING_URI}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXP_NAME)

    with mlflow.start_run(tags={"dataset_version": dataset_version}) as run:

        mlflow.log_params({
            "dataset_version": dataset_version,
            "model":           MODEL_WEIGHTS,
            "epochs":          epochs,
            "imgsz":           imgsz,
            "batch":           batch,
        })

        metrics_map = {
            "precision_box":  "metrics/precision(B)",
            "recall_box":     "metrics/recall(B)",
            "mAP50_box":      "metrics/mAP50(B)",
            "mAP50_95_box":   "metrics/mAP50-95(B)",
            "precision_mask": "metrics/precision(M)",
            "recall_mask":    "metrics/recall(M)",
            "mAP50_mask":     "metrics/mAP50(M)",
            "mAP50_95_mask":  "metrics/mAP50-95(M)",
            "fitness":        "fitness",
        }

        logged_metrics = {}
        logged_count = 0
        for mlflow_key, yolo_key in metrics_map.items():
            value = results.results_dict.get(yolo_key)
            if value is not None:
                mlflow.log_metric(mlflow_key, value)
                logged_metrics[mlflow_key] = round(float(value), 4)
                logged_count += 1
            else:
                log.warning(f"Metric '{yolo_key}' not found in results, skipping.")

        log.info(f"Logged {logged_count}/{len(metrics_map)} metrics to MLflow.")

        mlflow.log_artifact(str(best_weights), artifact_path="weights")
        log.info("Model artifact uploaded to S3 via MLflow.")

        run_id = run.info.run_id
        log.info(f"MLflow run complete. Run ID: {run_id}")

    log.info("MLflow logging complete.")
    return run_id, logged_metrics


# ── S3 model promotion ────────────────────────────────────────────────────────

def promote_model_to_s3(local_model_path: Path, dataset_version: str, model_name: str) -> dict:
    """
    Upload model to S3 under two keys:
      - crack-seg/<full_gitsha>/<model_name>   (immutable, versioned)
      - crack-seg/latest/best.pt               (mutable, always points to newest)
    Returns dict with both S3 keys.
    """
    s3 = boto3.client("s3")

    versioned_key = f"crack-seg/{dataset_version}/{model_name}"
    latest_key    = "crack-seg/latest/best.pt"

    log.info(f"Uploading model to s3://{MODELS_BUCKET}/{versioned_key} ...")
    s3.upload_file(str(local_model_path), MODELS_BUCKET, versioned_key)

    log.info(f"Copying model to s3://{MODELS_BUCKET}/{latest_key} ...")
    s3.copy_object(
        Bucket=MODELS_BUCKET,
        CopySource={"Bucket": MODELS_BUCKET, "Key": versioned_key},
        Key=latest_key,
    )

    log.info("Model promoted to S3 (versioned + latest).")
    return {
        "versioned_key": versioned_key,
        "latest_key":    latest_key,
    }


# ── Whylogs baseline ──────────────────────────────────────────────────────────

def build_and_upload_baseline(model_path: Path, dataset_version: str) -> str:
    """
    Run inference on all val images, build a whylogs profile,
    upload to S3. Returns the S3 key of the baseline profile.
    """
    val_images_dir = DATASET_PATH / "images" / "val"
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(val_images_dir.glob(ext)))

    if not image_files:
        log.warning(f"No validation images found in {val_images_dir}. Skipping baseline.")
        return ""

    log.info(f"Building whylogs baseline from {len(image_files)} validation images...")
    model = YOLO(str(model_path))
    records = []

    for img_path in image_files:
        try:
            results = model.predict(str(img_path), conf=0.30, iou=0.45, verbose=False)
            if not results:
                continue
            boxes = results[0].boxes
            num_detections = len(boxes) if boxes is not None else 0
            confidences    = boxes.conf.tolist() if boxes is not None and len(boxes) > 0 else []
            mean_conf      = sum(confidences) / len(confidences) if confidences else 0.0
            max_conf       = max(confidences) if confidences else 0.0
            records.append({
                "num_detections": num_detections,
                "mean_confidence": mean_conf,
                "max_confidence":  max_conf,
            })
        except Exception as e:
            log.warning(f"Skipping {img_path.name}: {e}")
            continue

    if not records:
        log.warning("No inference results collected. Skipping baseline upload.")
        return ""

    results_why  = why.log(records)
    profile      = results_why.profile()
    profile_path = Path("/tmp/baseline_profile.bin")
    profile.write(profile_path)

    s3_key = f"crack-seg/{dataset_version}/baseline_profile.bin"
    s3 = boto3.client("s3")
    s3.upload_file(str(profile_path), MODELS_BUCKET, s3_key)
    log.info(f"Baseline profile uploaded to s3://{MODELS_BUCKET}/{s3_key}")
    return s3_key


# ── Manifest generation & push ────────────────────────────────────────────────

def generate_and_push_manifest(
    dataset_version: str,
    model_s3_keys: dict,
    baseline_s3_key: str,
    mlflow_run_id: str,
    mlflow_metrics: dict,
    model_name: str,
):
    """
    Write model-manifest.json and push it to the app repo.
    This push triggers the CI pipeline automatically.
    """
    if not APP_REPO_URL or not APP_REPO_TOKEN:
        log.warning("APP_REPO_URL or APP_REPO_TOKEN not set — skipping manifest push.")
        return

    manifest = {
        "model_name":              model_name,
        "model_s3_key":            model_s3_keys["versioned_key"],
        "model_s3_key_latest":     model_s3_keys["latest_key"],
        "baseline_profile_s3_key": baseline_s3_key,
        "git_sha":                 dataset_version,
        "mlflow_run_id":           mlflow_run_id,
        "mlflow_metrics":          mlflow_metrics,
        "models_bucket":           MODELS_BUCKET,
        "trained_at":              datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = Path("/tmp/model-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"Manifest generated:\n{json.dumps(manifest, indent=2)}")

    # Clone app repo, update manifest, push
    clone_dir = Path("/tmp/app-repo")
    if clone_dir.exists():
        subprocess.run(["rm", "-rf", str(clone_dir)], check=True)

    # Build authenticated URL
    repo_url_auth = APP_REPO_URL.replace(
        "https://", f"https://x-access-token:{APP_REPO_TOKEN}@"
    )

    log.info("Cloning app repo...")
    subprocess.run(["git", "clone", repo_url_auth, str(clone_dir)], check=True)

    # Copy manifest into cloned repo
    import shutil
    shutil.copy(str(manifest_path), str(clone_dir / "model-manifest.json"))

    # Git commit and push
    subprocess.run(["git", "config", "user.email", "github-actions@github.com"], cwd=clone_dir, check=True)
    subprocess.run(["git", "config", "user.name",  "CT Pipeline"],               cwd=clone_dir, check=True)
    subprocess.run(["git", "add",    "model-manifest.json"],                      cwd=clone_dir, check=True)

    diff = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=clone_dir)
    if diff.returncode == 0:
        log.info("model-manifest.json unchanged — no commit needed.")
        return

    commit_msg = f"chore: update model-manifest.json [CT {dataset_version[:7]}]"
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=clone_dir, check=True)
    subprocess.run(["git", "push"],                     cwd=clone_dir, check=True)
    log.info("model-manifest.json pushed to app repo. CI pipeline will trigger.")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int, default=5)
    parser.add_argument("--imgsz",           type=int, default=640)
    parser.add_argument("--batch",           type=int, default=16)
    parser.add_argument("--workers",         type=int, default=4)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-dvc-pull",   action="store_true")
    args = parser.parse_args()

    dataset_version = get_dvc_dataset_version()
    model_name      = get_model_name(dataset_version)

    log.info(f"MLflow server  : {MLFLOW_TRACKING_URI}")
    log.info(f"Dataset version: {dataset_version}")
    log.info(f"Model filename : {model_name}")

    if not args.skip_dvc_pull:
        pull_dataset()
    if not args.skip_validation:
        validate_dataset()

    best_weights, results = train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        model_name=model_name,
    )

    # 1. Log to MLflow
    mlflow_run_id, mlflow_metrics = log_to_mlflow(
        results=results,
        best_weights=best_weights,
        dataset_version=dataset_version,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )

    # 2. Promote model to S3 (versioned + latest)
    model_s3_keys = promote_model_to_s3(
        local_model_path=best_weights,
        dataset_version=dataset_version,
        model_name=model_name,
    )

    # 3. Build whylogs baseline from val images and upload to S3
    baseline_s3_key = build_and_upload_baseline(
        model_path=best_weights,
        dataset_version=dataset_version,
    )

    # 4. Generate manifest and push to app repo (triggers CI)
    generate_and_push_manifest(
        dataset_version=dataset_version,
        model_s3_keys=model_s3_keys,
        baseline_s3_key=baseline_s3_key,
        mlflow_run_id=mlflow_run_id,
        mlflow_metrics=mlflow_metrics,
        model_name=model_name,
    )

    log.info(f"Training pipeline complete. Model: {model_name}")


if __name__ == "__main__":
    main()
