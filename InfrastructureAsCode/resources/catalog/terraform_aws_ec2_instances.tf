## A Terraform file that allows to create a number of VM instances in AWS EC2.
## This Terraform file produces as output files: (i) the SSH private key and (ii) the DNS of the created VM instances, thus enabling to access them via SSH.

# Configure the AWS Provider
provider "aws" {
  region     = "eu-west-3"
  access_key = var.aws_access_key_id
  secret_key = var.aws_secret_access_key
}

# Declare Terraform variables
variable "aws_access_key_id" {}
variable "aws_secret_access_key" {}
variable "aws_private_key" {}
variable "aws_keypair_name" {}
variable "aws_instances_count" {}
variable "aws_ami" {}
variable "aws_instance_type" {}

# Create a private SSH key 
resource "tls_private_key" "activeeon_private_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

# Create a SSH keypair (based on the private key mentioned above) 
resource "aws_key_pair" "activeeon_keypair" {
  key_name   = var.aws_keypair_name
  public_key = tls_private_key.activeeon_private_key.public_key_openssh
}

# Create a number of VM instances in AWS EC2 
resource "aws_instance" "activeeon_aws_ec2_instances" {
  count         = var.aws_instances_count
  ami           = var.aws_ami
  instance_type = var.aws_instance_type
  key_name      = aws_key_pair.activeeon_keypair.key_name
}

# Store the private key locally to be used to access VM instances via SSH
resource "local_file" "private_key" {
  filename        = "${path.module}/artefacts/${var.aws_private_key}"
  content         = tls_private_key.activeeon_private_key.private_key_pem
  file_permission = "0777"
}

# Store the public DNS of each VM locally (to be used for SSH access)
resource "local_file" "activeeon_aws_ec2_instance_dns" {
  count           = var.aws_instances_count
  filename        = "${path.module}/artefacts/aws_instance_${count.index}"
  file_permission = "0777"
  content         = <<EOF
${aws_instance.activeeon_aws_ec2_instances[count.index].public_ip}
  EOF
}

