output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.training.id
}

output "ami_used" {
  description = "Deep Learning AMI used for training"
  value       = data.aws_ami.deep_learning.name
}