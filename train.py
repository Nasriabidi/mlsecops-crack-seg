
import sys
import subprocess
import argparse
import logging
from datetime import datetime
from pathlib import Path

import mlflow
from ultralytics import YOLO

DATASET_PATH    = Path("./crack-seg")
DATASET_YAML    = "crack-seg.yaml"
MODEL_WEIGHTS   = "yolov8n-seg.pt"
MLFLOW_S3_URI   = "s3://mlsecops-mlflow-351611731527"
MLFLOW_EXP_NAME = "crack-seg-training"

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

    best_weights = Path("runs/segment/crack_seg/weights/best.pt")
    if not best_weights.exists():
        log.error(f"Training finished but best.pt not found at {best_weights}")
        sys.exit(1)

    # Rename best.pt → crack_seg_<sha7>_<YYYYMMDD>.pt
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
):
    log.info("Logging experiment to MLflow...")

    mlflow.set_tracking_uri(MLFLOW_S3_URI)
    mlflow.set_experiment(MLFLOW_EXP_NAME)

    with mlflow.start_run():

        # ── Params ────────────────────────────────────────────────────────────
        mlflow.log_params({
            "dataset_version": dataset_version,
            "model":           MODEL_WEIGHTS,
            "epochs":          epochs,
            "imgsz":           imgsz,
            "batch":           batch,
        })

        # ── Metrics ───────────────────────────────────────────────────────────
        metrics_map = {
            "mAP50":    "metrics/mAP50(M)",
            "mAP50_95": "metrics/mAP50-95(M)",
            "box_loss": "val/box_loss",
            "seg_loss": "val/seg_loss",
        }

        for mlflow_key, yolo_key in metrics_map.items():
            value = results.results_dict.get(yolo_key)
            if value is not None:
                mlflow.log_metric(mlflow_key, value)
            else:
                log.warning(f"Metric '{yolo_key}' not found in results, skipping.")

        # ── Artifact: named model file ────────────────────────────────────────
        mlflow.log_artifact(str(best_weights), artifact_path="weights")

        run_id = mlflow.active_run().info.run_id
        log.info(f"MLflow run logged. Run ID: {run_id}")

    log.info("MLflow logging complete.")


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

    log.info(f"Dataset version (Git SHA): {dataset_version}")
    log.info(f"Model will be saved as:    {model_name}")

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

    log_to_mlflow(
        results=results,
        best_weights=best_weights,
        dataset_version=dataset_version,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )

    log.info(f"Training pipeline complete. Model: {model_name}")


if __name__ == "__main__":
    main()