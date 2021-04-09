import requests

global variables


def raiser(msg): raise Exception(msg)


# Get variables
model_name = variables.get("MODEL_NAME") \
    if variables.get("MODEL_NAME") else raiser("MODEL_NAME should be specified")
model_version = variables.get("MODEL_VERSION") \
    if variables.get("APPEND") else None
api_undeploy = variables.get("UNDEPLOY_ENDPOINT") \
    if variables.get("UNDEPLOY_ENDPOINT") else raiser(
    "UNDEPLOY_ENDPOINT is None")
service_token = variables.get("SERVICE_TOKEN") \
    if variables.get("SERVICE_TOKEN") else variables.get("SERVICE_TOKEN_PROPAGATED")
service_endpoint = variables.get("MaaS_DL_INSTANCE_ENDPOINT") \
    if variables.get("MaaS_DL_INSTANCE_ENDPOINT") else variables.get("ENDPOINT_MODEL")

api_undeploy_endpoint = service_endpoint + api_undeploy
print("[INFO] api_undeploy_endpoint: ", api_undeploy_endpoint)

data = {'api_token': service_token, 'model_name': model_name, 'model_version': model_version}

try:
    req = requests.post(api_undeploy_endpoint, data=data, verify=False)
    print(req.text)
except Exception as e:
    raiser(e)
