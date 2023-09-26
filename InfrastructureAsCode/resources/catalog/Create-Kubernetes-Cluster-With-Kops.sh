#!/bin/bash

# Display environment variables before starting the cluster creation
printenv

# Execute kops commands to create a Kubernetes cluster
kops create cluster  \
    --name=${NAME} \
    --cloud=aws \
    --zones=${ZONES} \
    --node-count=${NODE_COUNT} \
    --node-size=${NODE_SIZE} \
    --discovery-store=s3://${OIDC_BUCKET}/${NAME}/discovery \
    --ssh-public-key=/root/.ssh/id_rsa.pub \
    --cloud-labels="cluster=${NAME}" \
    --yes 
#    --dry-run --output yaml \
    
#kops update cluster --name ${NAME} --yes --admin
kops get clusters
kops validate cluster --wait 20m

kubectl get nodes --show-labels
kubectl -n kube-system get po

cp /root/.kube/config /root/.kube/${SHORT_NAME}-config
cp /root/.ssh/id_rsa /root/.ssh/${SHORT_NAME}-ssh-private-key

chmod go+r /root/.kube/${SHORT_NAME}-config
chmod go+r /root/.ssh/${SHORT_NAME}-ssh-private-key
