#!/bin/bash

# ── Variables injected by Terraform templatefile ──────────────────────────────
REPO_URL="${repo_url}"
GIT_SHA="${git_sha}"
MODELS_BUCKET="${models_bucket}"

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_FILE="/var/log/training.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo " MLSecOps CT - Training Script"
echo " Git SHA: $GIT_SHA"
echo " $(date -u)"
echo "=============================================="

# ── Upload log to S3 on exit ──────────────────────────────────────────────────
upload_log() {
  echo "Uploading log to S3... $(date -u)"
  aws s3 cp "$LOG_FILE" \
    "s3://$MODELS_BUCKET/crack-seg/$GIT_SHA/training.log" || echo "Log upload failed"
}
trap upload_log EXIT

# ── System update + dependencies ──────────────────────────────────────────────
echo "[1/6] Installing system dependencies... $(date -u)"
apt-get update -q
apt-get install -y git python3-pip libgl1 libglib2.0-0
echo "[1/6] DONE"

# ── Clone repo at exact commit ────────────────────────────────────────────────
echo "[2/6] Cloning repo at commit $GIT_SHA... $(date -u)"
cd /home/ubuntu
git clone "$REPO_URL" training || { echo "ERROR: git clone failed"; exit 1; }
cd training
git checkout "$GIT_SHA" || { echo "ERROR: git checkout failed"; exit 1; }
echo "[2/6] DONE"

# ── Install Python dependencies ───────────────────────────────────────────────
echo "[3/6] Installing Python dependencies... $(date -u)"
pip3 install ultralytics mlflow boto3 Pillow --quiet || { echo "ERROR: pip install failed"; exit 1; }
pip3 install awscli --quiet --upgrade || echo "WARNING: awscli upgrade failed, continuing"
pip3 install dvc --quiet || { echo "ERROR: dvc install failed"; exit 1; }
pip3 install dvc-s3 --quiet || echo "WARNING: dvc-s3 install failed, continuing"
pip3 install s3fs --quiet || echo "WARNING: s3fs install failed, continuing"
echo "[3/6] DONE"

# ── Run training pipeline ─────────────────────────────────────────────────────
echo "[4/6] Starting training pipeline... $(date -u)"
python3 train.py \
  --epochs 3 \
  --imgsz 320 \
  --batch 64 \
  --workers 8
TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
  echo "ERROR: Training failed with exit code $TRAIN_EXIT"
  exit 1
fi
echo "[4/6] DONE"

# ── Find the named model file ─────────────────────────────────────────────────
echo "[5/6] Locating trained model... $(date -u)"
MODEL_FILE=$(find runs/ -name "crack_seg_*.pt" | head -1)

if [ -z "$MODEL_FILE" ]; then
  echo "ERROR: No model file found. Training may have failed."
  exit 1
fi

MODEL_NAME=$(basename "$MODEL_FILE")
echo "Model found: $MODEL_NAME"
echo "[5/6] DONE"

# ── Push model to S3 ──────────────────────────────────────────────────────────
echo "[6/6] Pushing model to S3... $(date -u)"
aws s3 cp "$MODEL_FILE" \
  "s3://$MODELS_BUCKET/crack-seg/$GIT_SHA/$MODEL_NAME" || { echo "ERROR: S3 upload failed"; exit 1; }

echo "=============================================="
echo " Model pushed to:"
echo " s3://$MODELS_BUCKET/crack-seg/$GIT_SHA/$MODEL_NAME"
echo " $(date -u)"
echo "=============================================="
echo "[6/6] DONE"

echo "Training complete. Instance will shut down now."
shutdown -h now