import os
import sys
import uuid

import requests
import wget

global variables


def raiser(msg): raise Exception(msg)

# Get variables
model_name = variables.get("MODEL_NAME") \
    if variables.get("MODEL_NAME") else raiser("MODEL_NAME should be specified")
model_version = variables.get("MODEL_VERSION") \
    if variables.get("MODEL_VERSION") else None
api_predict = variables.get("PREDICT_ENDPOINT") \
    if variables.get("PREDICT_ENDPOINT") else raiser("PREDICT_ENDPOINT is None")
service_token = variables.get("SERVICE_TOKEN") \
    if variables.get("SERVICE_TOKEN") else variables.get("SERVICE_TOKEN_PROPAGATED")
service_endpoint = variables.get("MaaS_DL_INSTANCE_ENDPOINT") \
    if variables.get("MaaS_DL_INSTANCE_ENDPOINT") else variables.get("ENDPOINT_MODEL")
instances = variables.get("INSTANCES") \
    if variables.get("INSTANCES") else variables.get("PROPAGATED_INSTANCES")
class_names = variables.get("CLASS_NAMES") \
    if variables.get("CLASS_NAMES") else variables.get("PROPAGATED_CLASS_NAMES")

api_predict_endpoint = service_endpoint + api_predict
print("[INFO] api_predict_endpoint: ", api_predict_endpoint)

data = {'api_token': service_token, 'model_name': model_name, 'model_version': model_version, 'instances': instances, 'class_names':class_names}

try:
    req = requests.post(api_predict_endpoint, data=data, verify=False)
    print(req.text)
except Exception as e:
    raiser(e)