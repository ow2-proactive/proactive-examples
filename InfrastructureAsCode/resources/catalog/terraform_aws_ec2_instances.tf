# Configure the AWS Provider
provider "aws" {
  region     = "eu-west-3"
  access_key = var.aws_access_key_id
  secret_key = var.aws_secret_access_key
}

variable "aws_access_key_id" {}
variable "aws_secret_access_key" {}
variable "aws_private_key" {}
variable "aws_keypair_name" {}
variable "aws_instances_count" {}
variable "aws_ami" {}
variable "aws_instance_type" {}


resource "tls_private_key" "activeeon_private_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "activeeon_keypair" {
  key_name   = var.aws_keypair_name
  public_key = tls_private_key.activeeon_private_key.public_key_openssh
}

resource "aws_instance" "activeeon_aws_ec2_instances" {
  count         = var.aws_instances_count
  ami           = var.aws_ami
  instance_type = var.aws_instance_type
  key_name      = aws_key_pair.activeeon_keypair.key_name
}

resource "local_file" "private_key" {
  filename        = "${path.module}/artefacts/${var.aws_private_key}"
  content         = tls_private_key.activeeon_private_key.private_key_pem
  file_permission = "0777"
}

resource "local_file" "activeeon_aws_ec2_instance_dns" {
  count           = var.aws_instances_count
  filename        = "${path.module}/artefacts/aws_instance_dns_${count.index}"
  file_permission = "0777"
  content         = <<EOF
${aws_instance.activeeon_aws_ec2_instances[count.index].public_dns}
  EOF
}

