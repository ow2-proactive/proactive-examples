#!/bin/bash

################### CREATE KOPS state bucket in AWS  ################### 
STATE_BUCKET=$1

aws s3api head-bucket --bucket $STATE_BUCKET
if [ $? -eq 0 ]; then
    echo "$STATE_BUCKET already exists"
else
    aws s3api create-bucket --bucket $STATE_BUCKET --region $variables_AWS_REGION --create-bucket-configuration LocationConstraint=$variables_AWS_REGION
    aws s3api put-bucket-encryption  --bucket $STATE_BUCKET  --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
    aws s3 ls | grep $STATE_BUCKET
fi

################### CREATE KOPS OIDC bucket in AWS  ################### 
OIDC_BUCKET=$2

aws s3api head-bucket --bucket $OIDC_BUCKET
if [ $? -eq 0 ]; then
    echo "$OIDC_BUCKET already exists"
else
    aws s3api create-bucket --bucket $OIDC_BUCKET --region $variables_AWS_REGION --object-ownership BucketOwnerPreferred --create-bucket-configuration LocationConstraint=$variables_AWS_REGION
    aws s3api put-public-access-block --bucket $OIDC_BUCKET --public-access-block-configuration BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false
    aws s3 sync s3://$OIDC_BUCKET . --acl public-read
    aws s3api put-bucket-acl --bucket $OIDC_BUCKET --acl public-read
fi
