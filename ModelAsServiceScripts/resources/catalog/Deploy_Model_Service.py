import os, sys, bz2, uuid, json, time
import subprocess


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Install required Python libraries if they are not already installed
try:
    if sys.version_info[0] < 3:
        from urllib import unquote
    else:
        from urllib.parse import unquote
except ImportError:
    install('request')
    if sys.version_info[0] < 3:
        from urllib import unquote
    else:
        from urllib.parse import unquote

try:
    import pickle
except ImportError:
    install('pickle')
    import pickle

try:
    import wget
except ImportError:
    install('wget')
    import wget

try:
    import requests
except ImportError:
    install('requests')
    import requests


# Define failure procedure that will be executed if something goes wrong
def failure(error_msg):
    current_status = "ERROR"
    variables.put("CURRENT_STATUS", current_status)
    print(error_msg)
    print("CURRENT_STATUS: ", current_status)
    sys.exit(0)


# Set default variables
current_status = "RUNNING"
model_path = os.path.join(os.getcwd(), "model.pkl")
baseline_data_path = os.path.join(os.getcwd(), "baseline_data.csv")

# Edit these variables if they don't correspond to your service endpoints
api_token = "/api/get_token"
api_deploy = "/api/deploy"

# Save Status File
instance_name = variables.get("INSTANCE_NAME")
file_name = instance_name + "_status"
file = open(file_name, "w")
file.write(instance_name)
file.close()

# Get variables
user_name = variables.get("USER_NAME") if variables.get(
    "USER_NAME") else failure("USER_NAME must be defined!")
api_endpoint = variables.get("API_ENDPOINT")
api_token_endpoint = api_endpoint + api_token
api_deploy_endpoint = api_endpoint + api_deploy
print("USER_NAME: ", user_name)
print("API_TOKEN_ENDPOINT:  ", api_token_endpoint)
print("API_DEPLOY_ENDPOINT: ", api_deploy_endpoint)

# Get a token to be able to deploy the model
try:
    data = {'user': user_name}
    req = requests.post(api_token_endpoint, data=data, verify=False)
    token = req.text
    if token.lower() == "Invalid user".lower():
        failure("USER_NAME is invalid")
    print("TOKEN: ", token)
except requests.exceptions.RequestException as e:
    failure(e)

# Download the model that will be deployed
model_url = variables.get("MODEL_URL") if variables.get(
    "MODEL_URL") else failure("MODEL_URL must be defined!")
model_url = unquote(model_url)
try:
    wget.download(model_url, model_path)
except Exception as e:
    failure(e)
    
baseline_data_url = variables.get("BASELINE_DATA_URL") if variables.get("BASELINE_DATA_URL") else failure("Baseline data is not defined!")
baseline_data_url = unquote(baseline_data_url)
try:
	wget.download(baseline_data_url, baseline_data_path)
except Exception as e:
    failure(e)

# Deploy the downloaded model
model_file = open(model_path, 'rb')
baseline_data_file = open(baseline_data_path, 'r')
files = {'model_file': model_file, 'baseline_data' : baseline_data_file }
data = {'api_token': token}

model_metadata_json = variables.get("MODEL_METADATA") if variables.get(
    "MODEL_METADATA") else None
if model_metadata_json is not None:
    data['model_metadata_json'] = unquote(model_metadata_json)

deviation_detection = variables.get("DEVIATION_DETECTION")
if deviation_detection is not None and deviation_detection.lower() == "true":
    data['drift_enabled'] = True
    data['drift_notification'] = True
else:
    data['drift_enabled'] = False
    data['drift_notification'] = False

deviation_treshold = variables.get("DEVIATION_TRESHOLD")
deviation_treshold = float(
    deviation_treshold) if deviation_treshold is not None else 0
data['drift_threshold'] = deviation_treshold

logging_prediction = variables.get("LOGGING_PREDICTION")
if logging_prediction is not None and logging_prediction.lower() == "true":
    data['log_predictions'] = True
else:
    data['log_predictions'] = False

try:
    req = requests.post(
        api_deploy_endpoint, files=files, data=data, verify=False)
    model_deployment_status = req.text
    if model_deployment_status.lower() != "Model deployed".lower():
        current_status = "ERROR"
    print("MODEL DEPLOYMENT STATUS: ", model_deployment_status)
except Exception as e:
    failure(e)
finally:
    model_file.close()

# Propagate the status of the deployment to the Post_Scrpit
variables.put("CURRENT_STATUS", current_status)