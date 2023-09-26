#!/bin/bash

USER_NAME=$1
GROUP_NAME=$2

aws iam get-user --user-name $USER_NAME
if [ $? -eq 0 ]; then
    echo "User $USER_NAME already exists"
else
    aws iam get-group --group-name $GROUP_NAME
    if [ $? -eq 0 ]; then
        echo "Group $GROUP_NAME already exists"
    else
    	aws iam create-group --group-name $GROUP_NAME
    	aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess --group-name $GROUP_NAME
    	aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonRoute53FullAccess --group-name $GROUP_NAME
    	aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess --group-name $GROUP_NAME
    	aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/IAMFullAccess --group-name $GROUP_NAME
    	aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonVPCFullAccess --group-name $GROUP_NAME
    	aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess --group-name $GROUP_NAME
    	aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonEventBridgeFullAccess --group-name $GROUP_NAME
    	aws iam create-user --user-name $USER_NAME
    	aws iam add-user-to-group --user-name $USER_NAME --group-name $GROUP_NAME
    	aws iam create-access-key --user-name $USER_NAME
    fi
fi
