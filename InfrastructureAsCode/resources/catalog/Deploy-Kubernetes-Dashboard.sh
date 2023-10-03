#!/bin/bash


NAME=$1
SHORT_NAME=$2
TOKEN_DURATION=$3

# Create kubernetes dashboard
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml
kubectl create serviceaccount $NAME --namespace kubernetes-dashboard
kubectl create clusterrolebinding kubernetes-admin-role-binding --clusterrole=cluster-admin --user=system:serviceaccount:kubernetes-dashboard:$NAME
kubectl --namespace kubernetes-dashboard patch svc kubernetes-dashboard -p '{"spec": {"type": "LoadBalancer"}}'
  
# Wait for the dashboard deployment to finish
ATTEMPTS=0
LB_EXTERNAL_ADDRESS=""

until [ $ATTEMPTS -gt 36 ]
do
	LB_EXTERNAL_ADDRESS=$(kubectl get svc -n kubernetes-dashboard kubernetes-dashboard -o=jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    echo "Dashboard service load balancer address: $LB_EXTERNAL_ADDRESS" 
    
    if [ ! -z "$LB_EXTERNAL_ADDRESS" ] && [ "$LB_EXTERNAL_ADDRESS" != "none" ] && [ "$LB_EXTERNAL_ADDRESS" != "pending" ]; then
        break
    fi 

    sleep 10
    ((ATTEMPTS++))
done

if [ ! -z "$LB_EXTERNAL_ADDRESS" ]; then
	echo "https://$LB_EXTERNAL_ADDRESS" > /dashboard/$SHORT_NAME-dashboard-address
fi

TOKEN=$(kubectl -n kubernetes-dashboard create token $NAME --duration=$TOKEN_DURATION)
echo "$TOKEN" > /dashboard/$SHORT_NAME-dashboard-token

kubectl config set-credentials $NAME --token="$TOKEN"

cp /root/.kube/config /root/.kube/${SHORT_NAME}-config

chmod go+r /root/.kube/${SHORT_NAME}-config
chmod -R go+r /dashboard
