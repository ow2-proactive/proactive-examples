import os, sys, bz2, uuid, requests, json, pickle, wget, time

def raiser(msg): raise Exception(msg)

MODEL_PATH = "last_model.pkl"

# Get variables
API_DEPLOY = variables.get("API_EXTENSION") if variables.get("API_EXTENSION") else raiser("API_EXTENSION is None")
SERVICE_TOKEN = variables.get("SERVICE_TOKEN") if variables.get("SERVICE_TOKEN") else variables.get("SERVICE_TOKEN_PROPAGATED")
API_ENDPOINT = variables.get("DEPLOY_MODEL_ENDPOINT") if variables.get("DEPLOY_MODEL_ENDPOINT") else variables.get("ENDPOINT_MODEL")
API_DEPLOY_ENDPOINT = API_ENDPOINT + API_DEPLOY
print("API_DEPLOY_ENDPOINT: ", API_DEPLOY_ENDPOINT)

# load the model
input_variables = {'task.model_id': None}
for key in input_variables.keys():
    for res in results:
        value = res.getMetadata().get(key)
        if value is not None:
            input_variables[key] = value
            break

model_id = input_variables['task.model_id']

if variables.get(model_id) is not None:
    model_compressed = variables.get(model_id)
    model_bin = bz2.decompress(model_compressed)
    with open(MODEL_PATH, "wb") as f:
        model = f.write(model_bin)
    print('model size (original):   ', sys.getsizeof(MODEL_PATH), " bytes")
    MODEL_PATH = os.path.join(os.getcwd(), MODEL_PATH)
else:
    MODEL_URL = variables.get("MODEL_URL") if variables.get("MODEL_URL") else raiser("MODEL_URL is None")
    MODEL_PATH = os.path.join(os.getcwd(), MODEL_PATH)
    wget.download(MODEL_URL, MODEL_PATH)

final_model = open(MODEL_PATH, 'rb')
files = {'modelfile': final_model}
data = {'api_token': SERVICE_TOKEN}
try:
    req = requests.post(API_DEPLOY_ENDPOINT, files=files, data=data)
    print(req.text)
finally:
    final_model.close()
