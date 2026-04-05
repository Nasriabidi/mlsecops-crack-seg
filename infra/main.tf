data "aws_caller_identity" "current" {}

# ── Get latest Ubuntu 22.04 AMI ───────────────────────────────────────────────
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]  # Canonical official

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ── Default VPC ───────────────────────────────────────────────────────────────
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ── Security group — outbound only ────────────────────────────────────────────
resource "aws_security_group" "training" {
  name        = "mlsecops-training-sg"
  description = "Training EC2 - outbound only"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "mlsecops-training-sg" }
}

# ── IAM role for EC2 (S3 access without hardcoded credentials) ────────────────
resource "aws_iam_role" "ec2_training" {
  name = "ec2-training-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "ec2_s3" {
  name = "ec2-s3-policy"
  role = aws_iam_role.ec2_training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadDatasets"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.datasets_bucket}",
          "arn:aws:s3:::${var.datasets_bucket}/*"
        ]
      },
      {
        Sid    = "WriteModels"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:HeadObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.models_bucket}",
          "arn:aws:s3:::${var.models_bucket}/*"
        ]
      },
      {
        Sid    = "MLflowFullAccess"
        Effect = "Allow"
        Action = ["s3:*"]
        Resource = [
          "arn:aws:s3:::${var.mlflow_bucket}",
          "arn:aws:s3:::${var.mlflow_bucket}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2_training" {
  name = "ec2-training-profile"
  role = aws_iam_role.ec2_training.name
}

# ── EC2 training instance ─────────────────────────────────────────────────────
resource "aws_instance" "training" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.training.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_training.name

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }
  
  user_data = templatefile("${path.module}/../scripts/user_data.sh", {
    repo_url          = var.repo_url
    git_sha           = var.git_sha
    models_bucket     = var.models_bucket
    mlflow_server_url = var.mlflow_server_url
  })

  tags = { Name = "mlsecops-training-${var.git_sha}" }
}