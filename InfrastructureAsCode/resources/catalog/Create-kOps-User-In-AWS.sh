#!/bin/bash


USER_NAME=$1
GROUP_NAME=$2

USER_EXISTS=1
aws iam get-user --user-name $USER_NAME || USER_EXISTS=0

if [ $USER_EXISTS -eq 1 ]; then
    echo "User $USER_NAME already exists"
else

    GROUP_EXISTS=1
    aws iam get-group --group-name $GROUP_NAME || GROUP_EXISTS=0
    
    if [ $GROUP_EXISTS -eq 0 ]; then
        echo "Creating kOps group $GROUP_NAME ..."
        aws iam create-group --group-name $GROUP_NAME
    	aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess --group-name $GROUP_NAME
    	aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonRoute53FullAccess --group-name $GROUP_NAME
    	aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess --group-name $GROUP_NAME
    	aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/IAMFullAccess --group-name $GROUP_NAME
    	aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonVPCFullAccess --group-name $GROUP_NAME
    	aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess --group-name $GROUP_NAME
    	aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonEventBridgeFullAccess --group-name $GROUP_NAME        
    fi
    echo "Creating kOps user $USER_NAME ..."
    aws iam create-user --user-name $USER_NAME
    aws iam add-user-to-group --user-name $USER_NAME --group-name $GROUP_NAME
    aws iam create-access-key --user-name $USER_NAME

fi
