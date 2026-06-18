# MLSecOps — Continuous Training Pipeline

> Part of a full MLSecOps pipeline built for automated and secure machine learning lifecycle management on AWS.

## Overview

This repository contains the **Continuous Training (CT)** pipeline for a crack detection model based on image segmentation. The pipeline automatically provisions cloud infrastructure, trains the model, tracks experiments, and promotes the best model to production — all triggered by a Git push.

## Model

- **Architecture:** YOLOv8n-seg (Ultralytics)
- **Task:** Instance segmentation for crack detection in construction surfaces
- **Dataset:** Crack segmentation dataset — 4029 images (flat format)

## Pipeline Architecture

```
Git Push → GitHub Actions → Terraform (EC2 c5.xlarge) → Training (YOLOv8n-seg)
        → MLflow Tracking → WhyLogs Baseline → Model Promotion (S3)
        → model-manifest.json → Triggers CI Pipeline
```

## Features

- **Infrastructure as Code** — EC2 training instance provisioned and destroyed automatically via Terraform
- **Experiment Tracking** — MLflow server on EC2 with Elastic IP and nginx reverse proxy
- **Data Versioning** — DVC with S3 remote (`mlsecops-datasets`)
- **Automatic Model Promotion** — best model pushed to S3 production path after training
- **Baseline Profiling** — WhyLogs baseline generated post-training for drift detection
- **Pipeline Linking** — `model-manifest.json` generated and pushed to app repo to trigger the CI pipeline

## Repository Structure

```
├── .github/
│   └── workflows/
│       └── ct-pipeline.yml       # GitHub Actions CT workflow
├── INFRA/
│   └── training/                 # Terraform modules for EC2 provisioning
│       ├── main.tf
│       ├── variables.tf
│       └── user_data.sh          # EC2 bootstrap script
├── train.py                      # Training script (YOLOv8n-seg + MLflow + WhyLogs)
├── requirements.txt
└── README.md
```

## Tech Stack

| Tool | Role |
|------|------|
| GitHub Actions | CI/CD orchestration |
| Terraform | EC2 provisioning and teardown |
| AWS EC2 (c5.xlarge) | Training compute |
| AWS S3 | Dataset storage, model registry |
| DVC | Dataset versioning |
| MLflow | Experiment tracking |
| YOLOv8n-seg | Model architecture |
| WhyLogs | Data profiling and baseline generation |

## How It Works

1. A push to `main` triggers the GitHub Actions workflow
2. Terraform provisions an EC2 `c5.xlarge` instance with a Deep Learning environment
3. The training script downloads the dataset from S3 via DVC, trains the model, and logs metrics to MLflow
4. The best model is promoted to a fixed S3 production path with a metadata file
5. A WhyLogs baseline profile is generated from the training data
6. A `model-manifest.json` is created (containing model S3 keys, MLflow run ID, metrics, baseline profile path) and pushed to the application repository to trigger the CI pipeline
7. The EC2 instance is destroyed automatically in the `finally` block


## Related Repositories

| Repository | Description |
|------------|-------------|
| [mlsecops-app](https://github.com/Nasriabidi/app) | Application repo — CI/CD and Kubernetes manifests |

## Author

**Nasreddine Abidi** — Final year engineering student at ISI (Institut Supérieur d'Informatique), internship at RAK (Rokn Alkairouan Technical Services), Dubai.

Academic supervisor: Mme Safa Rejichi | Professional supervisor: M. Aymen Gassoumi
