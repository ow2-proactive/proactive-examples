#!/usr/bin/env python3

import argparse
import json
import numbers
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
import connexion
import flask
import numpy as np
import pandas as pd
import psutil
import requests
import utils

from binascii import hexlify
from datetime import datetime as dt
from os.path import join, exists, isfile
from tempfile import TemporaryFile
from urllib.parse import quote
from cryptography.fernet import Fernet
from flask_cors import CORS


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    import proactive
except ImportError:
    install('proactive')
    import proactive


# Environment variables
INSTANCE_PATH = os.getenv('INSTANCE_PATH') if os.getenv('INSTANCE_PATH') is not None else "/model_as_service"
MODEL_SAVE_PATH = join(INSTANCE_PATH, "tmp")
PROMETHEUS_SERVICE_INTERNAL_PORT = os.getenv('PROMETHEUS_SERVICE_INTERNAL_PORT') if os.getenv(
    'PROMETHEUS_SERVICE_INTERNAL_PORT') is not None else "9091"
CONFIG_FILE_POLL_SECONDS = os.getenv('CONFIG_FILE_POLL_SECONDS') if os.getenv(
    'CONFIG_FILE_POLL_SECONDS') is not None else "30"
DEBUG_ENABLED = os.getenv('DEBUG_ENABLED') if os.getenv('DEBUG_ENABLED') is not None else False
TRACE_ENABLED = os.getenv('TRACE_ENABLED') if os.getenv('TRACE_ENABLED') is not None else True
HTTPS_ENABLED = os.getenv('HTTPS_ENABLED') if os.getenv('HTTPS_ENABLED') is not None else False

os.environ['REST_API_PORT'] = "8501"
os.environ['MODEL_CONFIG_FILE'] = join(INSTANCE_PATH, 'models.config')
os.environ['TENSORFLOW_SERVER_LOGS_FILE'] = join(INSTANCE_PATH, 'tensorflow_server.log')
os.environ['MODEL_SAVE_PATH'] = join(INSTANCE_PATH, MODEL_SAVE_PATH)

# General parameters
REST_API_PORT = 8501
PROMETHEUS_SERVICE_PORT = 9091
MODEL_CONFIG_FILE = join(INSTANCE_PATH, 'models.config')
TRACE_FILE = join(INSTANCE_PATH, 'trace.txt')  # default trace file
CONFIG_FILE = join(INSTANCE_PATH, 'config.json')  # default config file
PREDICTIONS_FILE = join(INSTANCE_PATH, 'predictions.csv')  # default predictions file

TOKENS = {
    'user': hexlify(os.urandom(16)).decode(),  # api key
    'test': hexlify(os.urandom(16)).decode()
}  # user tokens

LOG_PREDICTIONS = False
USER_KEY = os.getenv('USER_KEY')
assert USER_KEY is not None, "USER_KEY is required!"
USER_KEY = str(USER_KEY).encode()

# Decrypt user credentials
USER_DATA_FILE = join(INSTANCE_PATH, 'user_data.enc')
with open(USER_DATA_FILE, 'rb') as f:
    encrypted_data = f.read()
fernet = Fernet(USER_KEY)
decrypted_data = fernet.decrypt(encrypted_data)
message = decrypted_data.decode()
user_credentials = json.loads(message)

# Get proactive server url
proactive_rest = user_credentials['ciUrl']
proactive_url = proactive_rest[:-5]

# Check if there the configuration file exists
if not isfile(CONFIG_FILE):
    print("Generating the configuration file")
    config = {
        'DEBUG_ENABLED': DEBUG_ENABLED,
        'TRACE_ENABLED': TRACE_ENABLED,
        'LOG_PREDICTIONS': LOG_PREDICTIONS,
        'HTTPS_ENABLED': HTTPS_ENABLED
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)
    print("Done")


# ----- Helper functions ----- #


def get_config(param, default_value=None):
    # print("Loading parameters from the configuration file")
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    if param in config:
        return config[param]
    else:
        return default_value


def set_config(param, value):
    # print("Writing to the configuration file")
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    config[param] = value
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


def trace(message, token=""):
    if get_config('TRACE_ENABLED'):
        datetime_str = dt.today().strftime('%Y-%m-%d %H:%M:%S')
        with open(TRACE_FILE, "a") as f:
            f.write("%s|%s|%s\n" % (datetime_str, token, message))


def log(message, token=""):
    trace(message, token)
    if get_config('DEBUG_ENABLED'):
        datetime_str = dt.today().strftime('%Y-%m-%d %H:%M:%S')
        print(datetime_str, token, message)
    return message


def submit_workflow_from_catalog(bucket_name, workflow_name, workflow_variables={}, token=""):
    result = False
    try:
        log("[INFO] Connecting on " + proactive_url, token)
        gateway = proactive.ProActiveGateway(proactive_url, [])
        gateway.connect(
            username=user_credentials['ciLogin'],
            password=user_credentials['ciPasswd'])
        if gateway.isConnected():
            log("[INFO] Connected to " + proactive_url, token)
            try:
                log("[INFO] Submitting a workflow from the catalog", token)
                jobId = gateway.submitWorkflowFromCatalog(bucket_name, workflow_name, workflow_variables)
                assert jobId is not None
                assert isinstance(jobId, numbers.Number)
                workflow_path = bucket_name + "/" + workflow_name
                log("[INFO] Workflow " + workflow_path + " submitted successfully with jobID: " + str(jobId), token)
                result = True
            finally:
                log("[INFO] Disconnecting from " + proactive_url, token)
                gateway.disconnect()
                log("[INFO] Disconnected from " + proactive_url, token)
                gateway.terminate()
                log("[INFO] Connection finished from " + proactive_url, token)
        else:
            log("[INFO] Couldn't connect to " + proactive_url, token)
    except Exception as e:
        log("[ERROR] Error while connecting on " + proactive_url, token)
        log(str(e), token)
    return result


def submit_web_notification(message, token):
    return submit_workflow_from_catalog("notification-tools", "Web_Notification", {'MESSAGE': message}, token)


def auth_token(token):
    for user, key in TOKENS.items():
        if key == token:
            return True
    return False


def get_token_user(token):
    for user, key in TOKENS.items():
        if key == token:
            return user
    return None


# ----- REST API endpoints ----- #


def get_token_api(user) -> str:
    addr = connexion.request.remote_addr
    token = TOKENS.get(user)
    if not token:
        log('[ERROR] Invalid user: {u} ({a})'.format(u=user, a=addr))
        return "Invalid user"
    else:
        log('[INFO] {u} token is {t} ({a})'.format(u=user, t=token, a=addr))
        return token


def predict_api(api_token, model_name, instances) -> str:
    api_token = connexion.request.form['api_token']
    if auth_token(api_token):
        model_name = connexion.request.form['model_name']
        instances = connexion.request.form["instances"]
        instances = json.loads(instances)
        class_names = connexion.request.form['class_names']
        class_names = class_names.split(",")
        log("[INFO] Calling predict_api", api_token)

        try:
            model_version = int(connexion.request.form["model_version"])
        except:
            model_version = utils.get_newest_deployed_version(model_name)
            pass

        model_version_status = utils.check_model_name_version(model_name, model_version)

        if model_version_status == "version deployed":
            try:
                data = json.dumps({"signature_name": "serving_default", "instances": instances})
                headers = {"content-type": "application/json"}
                prediction_link = "http://localhost:8501/v1/models/" + model_name + "/versions/" + str(
                    model_version) + ":predict"
                json_response = requests.post(prediction_link, data=data, headers=headers)
                predictions = json.loads(json_response.text)['predictions']
                return class_names[np.argmax(predictions[0])]
            except Exception as e:
                return log(str(e), api_token)
        else:
            return log("[INFO] " + model_version_status, api_token)
    else:
        return log("[INFO] Invalid token", api_token)


def deploy_api(model_name, model_file) -> str:
    model_zip_path = os.environ['MODEL_SAVE_PATH'] + str(uuid.uuid4()) + ".zip"
    api_token = connexion.request.form["api_token"]
    model_name = connexion.request.form["model_name"]
    append = connexion.request.form["append"]
    model_file = connexion.request.files["model_file"]
    model_download_path = os.environ['MODEL_SAVE_PATH'] + "/" + model_name
    download_model = True

    # if model_version is empty, model_version will be set on None
    try:
        model_version = int(connexion.request.form["model_version"])
    except:
        model_version = None
        pass

    log("[INFO] Calling deploy_api", api_token)
    if auth_token(api_token):
        # Service Status Management
        tensorflow_model_server_down = True
        for proc in psutil.process_iter():
            pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time'])
            if pinfo['name'] == "tensorflow_model_server":
                log("[INFO] TensorFlow model server is already up", api_token)
                tensorflow_model_server_down = False
        if tensorflow_model_server_down:
            log("[INFO] Starting a new tensorflow_model_server", api_token)
            tf_server = subprocess.Popen(["tensorflow_model_server "
                                          "--rest_api_port=$REST_API_PORT "
                                          "--model_config_file_poll_wait_seconds=$CONFIG_FILE_POLL_SECONDS "
                                          "--model_config_file=$MODEL_CONFIG_FILE > $TENSORFLOW_SERVER_LOGS_FILE 2>&1"],
                                         stdout=subprocess.DEVNULL,
                                         shell=True,
                                         preexec_fn=os.setsid)

        # Model Versioning Management
        # if model_version was not specified, it will be set by default as "the latest model_version number + 1"
        if model_version is None:
            if not os.path.exists(model_download_path):
                os.makedirs(model_download_path)
                model_version = 1
            else:
                listOfFile = os.listdir(model_download_path)
                model_versions = []
                for file in listOfFile:
                    file_casted = file
                    try:
                        file_casted = int(file)
                        model_versions.append(file_casted)
                    except:
                        pass
                # check if the model directory is empty or not
                if not model_versions:
                    model_version = 1
                else:
                    model_versions.sort()
                    model_version = model_versions[-1] + 1
            log("[INFO] new version to be deployed : " + str(model_version), api_token)
        else:
            version_path = model_download_path + "/" + str(model_version)
            if os.path.isdir(version_path):
                download_model = False
                log("[WARN] This model version already exists. \
                    The uploaded model version will be ignored. The existing version will be deployed.",
                    api_token)

        # Model Downloading
        # if the specified model version doesn't exist in the directory,
        # the zip file uploaded by the user will be downloaded
        if download_model:
            version_path = model_download_path + "/" + str(model_version)
            log("[INFO] Downloading the new model in " + str(version_path), api_token)
            model_file.save(model_zip_path)
            with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
                zip_ref.extractall(version_path)
                os.remove(model_zip_path)

        # Model Deployment
        log("[INFO] Deploying the version " + str(model_version) + " of " + model_name, api_token)
        if append == "true":
            deployment_status = utils.append_version_model_service_config(model_name, model_download_path,
                                                                          model_version)
        else:
            deployment_status = utils.add_version_model_service_config(model_name, model_download_path, model_version)
        # print("The new tensorflow model file was deployed successfully at: ",os.environ['MODELS_PATH'], model_version)
        return log("[INFO]" + deployment_status, api_token)
    else:
        return log("[INFO] Invalid token", api_token)


def list_deployed_models_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("[INFO] Calling list_models_api", api_token)
    if auth_token(api_token):
        models_config_list = utils.read_config_file()
        log("[INFO] List of deployed models:\n" + str(models_config_list), api_token)
        return str(models_config_list)
    else:
        return log("[INFO] Invalid token", api_token)


def undeploy_model_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("[INFO] Calling undeploy_api", api_token)
    if auth_token(api_token):
        model_name = connexion.request.form["model_name"]
        # if model_version is empty, model_version will be set on None
        try:
            model_version = int(connexion.request.form["model_version"])
        except:
            model_version = None
            pass
        status = utils.delete_version_model_service_config(model_name, model_version)
        log("[INFO] Model removed:\n" + str(model_name), api_token)
        return log("[INFO]" + status, api_token)
    else:
        return log("[ERROR] Invalid token", api_token)


def redeploy_api() -> str:
    api_token = connexion.request.form["api_token"]

    if auth_token(api_token):
        # Service Status Management
        tensorflow_model_server_down = True
        for proc in psutil.process_iter():
            pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time'])
            if pinfo['name'] == "tensorflow_model_server":
                log("[INFO] TensorFlow model server is up", api_token)
                tensorflow_model_server_down = False
        if tensorflow_model_server_down:
            log("[INFO] Starting a new tensorflow_model_server", api_token)
            tf_server = subprocess.Popen(["tensorflow_model_server "
                                          "--rest_api_port=$REST_API_PORT "
                                          "--model_config_file_poll_wait_seconds=$CONFIG_FILE_POLL_SECONDS "
                                          "--model_config_file=$MODEL_CONFIG_FILE > $TENSORFLOW_SERVER_LOGS_FILE 2>&1"],
                                         stdout=subprocess.DEVNULL,
                                         shell=True,
                                         preexec_fn=os.setsid)
        try:
            model_version = int(connexion.request.form["model_version"])
        except:
            model_version = None
        pass
        model_name = connexion.request.form["model_name"]
        append = connexion.request.form["append"]
        model_download_path = os.environ['MODEL_SAVE_PATH'] + "/" + model_name

        # Model Versioning Management
        # if model_version was not specified, it will be set by default as "the latest model_version number + 1"
        if model_version is None:
            if not os.path.exists(model_download_path):
                deployment_status = "[ERROR] There is no model stored with this name" + model_name + ". please choose one of the saved models"
            else:
                listOfFile = os.listdir(model_download_path)
                model_versions = []
                for file in listOfFile:
                    file_casted = file
                    try:
                        file_casted = int(file)
                        model_versions.append(file_casted)
                    except:
                        pass
                # check if the model directory is empty or not
                if not model_versions:
                    model_version = 1
                else:
                    model_versions.sort()
                    model_version = model_versions[-1]
                    version_path = model_download_path + "/" + str(model_version)
            log("[INFO] the version that will be redeployed is : " + str(model_version), api_token)
        else:
            version_path = model_download_path + "/" + str(model_version)
            if not os.path.isdir(version_path):
                deployment_status = "[ERROR] This model version path doesn't exist: " + version_path + ". Please choose an already uploaded model version."
                return log(deployment_status, api_token)
        # Model Deployment
        log("[INFO] Redeploying the version " + str(model_version) + " of " + model_name, api_token)
        if append == "true":
            deployment_status = "[INFO] " + utils.append_version_model_service_config(model_name, version_path,
                                                                                      model_version)
        else:
            deployment_status = "[INFO] " + utils.add_version_model_service_config(model_name, version_path,
                                                                                   model_version)
        return log(deployment_status, api_token)
    else:
        return log("[ERROR] Invalid token ", api_token)


def download_model_config_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("[INFO] Downloading the model config file", api_token)
    if auth_token(api_token):
        if not os.path.isdir(MODEL_CONFIG_FILE):
            return log("[ERROR] The model config file was not found. \
                Deploy a model or upload a new model config file and try again", api_token)
        else:
            return flask.send_from_directory(directory=INSTANCE_PATH, filename='models.config', as_attachment=True)
    else:
        return log("[ERROR] Invalid token", api_token)


def upload_model_config_api() -> str:
    api_token = connexion.request.form["api_token"]
    model_config_file = connexion.request.files["model_config_file"]
    if auth_token(api_token):
        model_config_file.save(MODEL_CONFIG_FILE)
        return log("[INFO] Model config file was successfully uploaded", api_token)
    else:
        return log("[ERROR] Invalid token", api_token)


def list_saved_models(json_response) -> str:
    api_token = connexion.request.form["api_token"]
    log("[INFO] Displaying the list of downloaded models", api_token)
    if auth_token(api_token):
        json_response = connexion.request.form["json_response"]
        if json_response == "true":
            model_download_path = 'tree -J ' + os.environ['MODEL_SAVE_PATH']
        else:
            model_download_path = 'tree ' + os.environ['MODEL_SAVE_PATH']
        tree_model_download_path = os.popen(model_download_path).read()
        return tree_model_download_path
    else:
        return log("[ERROR] Invalid token", api_token)


def clean_saved_models(model_name) -> str:
    api_token = connexion.request.form["api_token"]
    log("[INFO] Cleaning the downloaded models", api_token)
    if auth_token(api_token):
        model_name = connexion.request.form["model_name"]
        model_path = os.environ['MODEL_SAVE_PATH'] + "/" + model_name
        try:
            model_version = int(connexion.request.form["model_version"])
        except:
            model_version = None
            pass
        if not utils.check_deployed_model_name_version(model_name, model_version):
            if not os.path.exists(model_path):
                clean_status = "[ERROR] Model folder " + str(model_path) + " doesn't exist."
            elif model_version is None:
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                    clean_status = "[INFO] Model folder " + str(model_path) + " was successfully deleted."
            else:
                model_version_path = model_path + "/" + str(model_version)
                if os.path.exists(model_version_path):
                    shutil.rmtree(model_version_path)
                    clean_status = "[INFO] Model version folder " + str(model_path) + "/" + str(
                        model_version) + " was successfully deleted."
                else:
                    clean_status = "[ERROR] Model version folder " + str(model_path) + "/" + str(
                        model_version) + " doesn't exist."
        else:
            if model_version is None:
                clean_status = "[ERROR] The model " + model_name + " is deployed. \
                To be able to delete it, you should undeploy it first."
            else:
                clean_status = "[ERROR] The version " + str(
                    model_version) + " of the model " + model_name + " is deployed. \
                    To be able to delete it, you should undeploy it first."
    else:
        return log("[ERROR] Invalid token", api_token)
    return log(clean_status, api_token)


def trace_preview_api(key) -> str:
    if USER_KEY == key.encode():
        if exists(TRACE_FILE) and isfile(TRACE_FILE):
            header = ["Date Time", "Token", "Traceability information"]
            with open(TRACE_FILE) as f, TemporaryFile("w+") as t:
                for line in f:
                    ln = len(line.strip().split("|"))
                    if ln < 3:
                        line = "||" + line
                    t.write(line)
                t.seek(0)
                trace_dataframe = pd.read_csv(t, sep='|', names=header, engine='python')
                trace_dataframe.fillna('', inplace=True)
            with open(CONFIG_FILE) as f:
                config = json.load(f)
            dataframe_config = pd.DataFrame.from_records([config])
            config_result = dataframe_config.to_html(escape=False, classes='table table-bordered',
                                                     justify='center', index=False)
            trace_result = trace_dataframe.to_html(escape=False, classes='table table-bordered table-striped',
                                                   justify='center', index=False)

            result = """
            <!DOCTYPE html>
            <html>
              <head>
                <meta charset="UTF-8">
                  <title>Audit & Traceability</title>
                  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
              </head>
                <body>
                  <p><a href='ui/#/default' target='_blank'>Click here to access Swagger UI</a></p>
                  <h2 class="text-center my-4" style="color:#003050;">Audit & Traceability</h2>
                  <div style="text-align:center;">{0}
                  <br/>
                  <br/>
                  <br/>
                  {1}
                  </div>
                </body></html>""".format(config_result, trace_result)
            # Add link to log predictions if enabled
            if get_config('LOG_PREDICTIONS'):
                result = "<p><a href='predictions_preview?key=" + quote(
                    key) + "' target='_blank'>Click here to visualize the predictions</a></p>" + result
            return result
        else:
            return log("[INFO] Trace file is empty", key)
    else:
        return log("[ERROR] Invalid key", key)


# ----- Main entry point ----- #
if __name__ == '__main__':
    # logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9090, help="set the port that will be used to deploy the service")
    args = parser.parse_args()
    app = connexion.FlaskApp(__name__, port=args.port, specification_dir=INSTANCE_PATH)
    CORS(app.app)
    app.add_api('dl_service-api.yaml', arguments={'title': 'Deep Learning Model Service'})

    app.run(debug=DEBUG_ENABLED)