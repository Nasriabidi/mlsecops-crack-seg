#!/bin/bash

# ── Write the actual training script ─────────────────────────────────────────
cat > /home/ubuntu/run_training.sh << 'TRAINING_SCRIPT'
#!/bin/bash

REPO_URL="__REPO_URL__"
GIT_SHA="__GIT_SHA__"
MODELS_BUCKET="__MODELS_BUCKET__"
MLFLOW_SERVER_URL="__MLFLOW_SERVER_URL__"

LOG_FILE="/var/log/training.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo " MLSecOps CT - Training Script"
echo " Git SHA: $GIT_SHA"
echo " MLflow:  $MLFLOW_SERVER_URL"
echo " $(date -u)"
echo "=============================================="

upload_log() {
  echo "Uploading log to S3... $(date -u)"
  aws s3 cp "$LOG_FILE" \
    "s3://$MODELS_BUCKET/crack-seg/$GIT_SHA/training.log" || echo "Log upload failed"
}
trap upload_log EXIT

echo "[1/6] Installing system dependencies... $(date -u)"
apt-get update -q
apt-get install -y git python3-pip libgl1 libglib2.0-0
echo "[1/6] DONE $(date -u)"

echo "[2/6] Cloning repo... $(date -u)"
cd /home/ubuntu
git clone "$REPO_URL" training || { echo "ERROR: git clone failed"; exit 1; }
cd training
git checkout "$GIT_SHA" || { echo "ERROR: git checkout failed"; exit 1; }
echo "[2/6] DONE $(date -u)"

echo "[3/6] Installing Python dependencies... $(date -u)"
pip3 install ultralytics mlflow boto3 Pillow --quiet || { echo "ERROR: pip install failed"; exit 1; }
pip3 install awscli --quiet --upgrade || echo "WARNING: awscli upgrade failed"
pip3 install dvc --quiet || { echo "ERROR: dvc install failed"; exit 1; }
pip3 install dvc-s3 --quiet || echo "WARNING: dvc-s3 failed"
pip3 install s3fs --quiet || echo "WARNING: s3fs failed"
echo "[3/6] DONE $(date -u)"

echo "[4/6] Starting training... $(date -u)"
MLFLOW_TRACKING_URI="$MLFLOW_SERVER_URL" python3 train.py \
  --epochs 3 \
  --imgsz 320 \
  --batch 64 \
  --workers 4
TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
  echo "ERROR: Training failed with exit code $TRAIN_EXIT"
  exit 1
fi
echo "[4/6] DONE $(date -u)"

echo "[5/6] Locating model... $(date -u)"
MODEL_FILE=$(find runs/ -name "crack_seg_*.pt" | head -1)
if [ -z "$MODEL_FILE" ]; then
  echo "ERROR: No model file found"
  exit 1
fi
MODEL_NAME=$(basename "$MODEL_FILE")
echo "Model: $MODEL_NAME"
echo "[5/6] DONE $(date -u)"

echo "[6/6] Pushing model to S3... $(date -u)"
aws s3 cp "$MODEL_FILE" \
  "s3://$MODELS_BUCKET/crack-seg/$GIT_SHA/$MODEL_NAME" || { echo "ERROR: S3 upload failed"; exit 1; }
echo "[6/6] DONE $(date -u)"

echo "Training complete. Shutting down."
shutdown -h now
TRAINING_SCRIPT

# ── Inject variables into the script ─────────────────────────────────────────
sed -i "s|__REPO_URL__|${repo_url}|g"                   /home/ubuntu/run_training.sh
sed -i "s|__GIT_SHA__|${git_sha}|g"                     /home/ubuntu/run_training.sh
sed -i "s|__MODELS_BUCKET__|${models_bucket}|g"         /home/ubuntu/run_training.sh
sed -i "s|__MLFLOW_SERVER_URL__|${mlflow_server_url}|g" /home/ubuntu/run_training.sh

chmod +x /home/ubuntu/run_training.sh

# ── Run training detached from cloud-init ────────────────────────────────────
nohup /home/ubuntu/run_training.sh &