#!/bin/bash
set -euo pipefail

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

# ── System update + dependencies ──────────────────────────────────────────────
echo "[1/6] Installing system dependencies..."
apt-get update -q
apt-get install -y git python3-pip awscli

# ── Clone repo at exact commit ────────────────────────────────────────────────
echo "[2/6] Cloning repo at commit $GIT_SHA..."
cd /home/ubuntu
git clone "$REPO_URL" training
cd training
git checkout "$GIT_SHA"

# ── Install Python dependencies ───────────────────────────────────────────────
echo "[3/6] Installing Python dependencies..."
pip install --quiet -r requirements.txt

# ── Install DVC S3 support ────────────────────────────────────────────────────
pip install --quiet "dvc[s3]"

# ── Run training pipeline ─────────────────────────────────────────────────────
echo "[4/6] Starting training pipeline..."
python train.py \
  --epochs 5 \
  --imgsz 640 \
  --batch 16 \
  --workers 16

# ── Find the named model file ─────────────────────────────────────────────────
echo "[5/6] Locating trained model..."
MODEL_FILE=$(find runs/segment/crack_seg/weights/ -name "crack_seg_*.pt" | head -1)

if [ -z "$MODEL_FILE" ]; then
  echo "ERROR: No model file found. Training may have failed."
  exit 1
fi

MODEL_NAME=$(basename "$MODEL_FILE")
echo "Model found: $MODEL_NAME"

# ── Push model to S3 ──────────────────────────────────────────────────────────
echo "[6/6] Pushing model to S3..."
aws s3 cp "$MODEL_FILE" \
  "s3://$MODELS_BUCKET/crack-seg/$GIT_SHA/$MODEL_NAME"

echo "=============================================="
echo " Model pushed to:"
echo " s3://$MODELS_BUCKET/crack-seg/$GIT_SHA/$MODEL_NAME"
echo " $(date -u)"
echo "=============================================="

# ── Signal success then shutdown ──────────────────────────────────────────────
echo "Training complete. Instance will shut down now."
shutdown -h now