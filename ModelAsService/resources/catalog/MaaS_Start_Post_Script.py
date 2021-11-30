import os, sys, bz2, uuid, json, time
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])

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
    import requests
except ImportError:
    install('requests')
    import requests

time.sleep(60) 

ENDPOINT_MODEL = variables.get("ENDPOINT_" + variables.get("INSTANCE_NAME")) 
ENDPOINT_MODEL = ENDPOINT_MODEL.split("/api")[0]

variables.put("ENDPOINT_MODEL", ENDPOINT_MODEL)
API_TOKEN_ENDPOINT = ENDPOINT_MODEL + "/api/get_token"
print("API_TOKEN_ENDPOINT: " + API_TOKEN_ENDPOINT)

USER_NAME = variables.get("USER_NAME")
print("USER_NAME: ", USER_NAME)

try:
    PARAMS = {'user': USER_NAME}
    req = requests.get(API_TOKEN_ENDPOINT, params=PARAMS, verify=False)
    token = req.text
    if token.lower() == "Invalid user".lower():
        failure("USER_NAME is invalid")
    print("TOKEN: ", token)
except requests.exceptions.RequestException as e:
    failure(e)

variables.put("SERVICE_TOKEN_PROPAGATED", token)
