#!/usr/bin/env python3

# import os, sys, bz2, uuid, pickle, json, connexion, wget
# import numpy as np
import os
import sys
import glob
import json
import connexion
import subprocess
import numbers
import pandas as pd

from cryptography.fernet import Fernet
from datetime import datetime as dt
from os.path import join, exists, isfile
from binascii import hexlify
from shutil import move
from flask_cors import CORS
from joblib import load
from flask import jsonify

from scipy.stats import norm
from scipy.stats import wasserstein_distance


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Install required Python libraries if they are not already installed
try:
    import proactive
except ImportError:
    install('proactive')
    import proactive


# Environment variables
DEBUG_ENABLED = True if (os.getenv('DEBUG_ENABLED') is not None and os.getenv('DEBUG_ENABLED').lower() == "true") else False
TRACE_ENABLED = True if (os.getenv('TRACE_ENABLED') is not None and os.getenv('TRACE_ENABLED').lower() == "true") else False
DRIFT_ENABLED = True if (os.getenv('DRIFT_ENABLED') is not None and os.getenv('DRIFT_ENABLED').lower() == "true") else False
DRIFT_THRESHOLD = float(os.getenv('DRIFT_THRESHOLD')) if os.getenv('DRIFT_ENABLED') is not None else None
HTTPS_ENABLED = True if (os.getenv('HTTPS_ENABLED') is not None and os.getenv('HTTPS_ENABLED').lower() == "true") else False
USER_KEY = os.getenv('USER_KEY')
assert USER_KEY is not None, "USER_KEY is required!"
USER_KEY = str(USER_KEY).encode()


# General parameters
APP_BASE_DIR = "/model_as_a_service"
UPLOAD_MODELS_FOLDER = "/model_as_a_service"  # model folder
MODEL_FILE_EXT = '.model'
META_FILE_EXT = '.meta'
CURRENT_MODEL_FILE = join(UPLOAD_MODELS_FOLDER, 'model_last' + MODEL_FILE_EXT)  # default model path
CURRENT_META_FILE = join(UPLOAD_MODELS_FOLDER, 'model_last' + META_FILE_EXT)    # default meta path
TRACE_FILE = join(UPLOAD_MODELS_FOLDER, 'trace.txt')  # default trace file
TOKENS = {
    'user': hexlify(os.urandom(16)).decode(),  # api key
    'test': hexlify(os.urandom(16)).decode()
}  # user tokens


# Decrypt user credentials
USER_DATA_FILE = join(APP_BASE_DIR, 'user_data.enc')
with open(USER_DATA_FILE, 'rb') as f:
    encrypted_data = f.read()
fernet = Fernet(USER_KEY)
decrypted_data = fernet.decrypt(encrypted_data)
message = decrypted_data.decode()
user_credentials = json.loads(message)


# Get proactive server url
proactive_rest = user_credentials['ciUrl']
proactive_url = proactive_rest[:-5]


# ----- Helper functions ----- #


def trace(message, token=""):
    if TRACE_ENABLED:
        datetime_str = dt.today().strftime('%Y-%m-%d %H:%M:%S')
        with open(TRACE_FILE, "a") as f:
            f.write("%s %s %s\n" % (datetime_str, token, message))


def log(message, token=""):
    trace(message, token)
    if DEBUG_ENABLED:
        datetime_str = dt.today().strftime('%Y-%m-%d %H:%M:%S')
        print(datetime_str, token, message)
    return message


def perform_drift_detection(predict_dataframe, token=""):
    log("calling perform_drift_detection", token)
    if exists(CURRENT_META_FILE) and isfile(CURRENT_META_FILE):
        # log("The model has an associated metadata")
        model_metadata = pd.read_pickle(CURRENT_META_FILE)
        log("model_metadata:\n" + str(model_metadata), token)
        # log("Calculating data drift measures", token)
        predict_mean = predict_dataframe.mean(axis=0)  # mean
        predict_std = predict_dataframe.std(axis=0)    # standard deviation
        # predict_metadata = pd.DataFrame({'Mean': predict_mean, 'Std': predict_std}).T
        predict_metadata = pd.DataFrame({0: predict_mean, 1: predict_std}).T
        log("predict_metadata:\n" + str(predict_metadata), token)
        size_data = len(model_metadata.columns)
        model_metadata_normal = norm.rvs(
            size=size_data,
            loc=model_metadata.iloc[0],    # mean
            scale=model_metadata.iloc[1])  # std
        predict_metadata_normal = norm.rvs(
            size=size_data,
            loc=predict_metadata.iloc[0],    # mean
            scale=predict_metadata.iloc[1])  # std
        # wasserstein distance
        score = wasserstein_distance(model_metadata_normal, predict_metadata_normal)
        log("Wasserstein distance: " + str(score), token)
        log("Drift threshold was set to: " + str(DRIFT_THRESHOLD), token)
        # Send web notification alerts
        if DRIFT_THRESHOLD is not None and score > DRIFT_THRESHOLD:
            log("Data drift detected!\nSending a web notification...", token)
            message = "MaaS data drift detected from " + get_token_user(token) + " (" + token + ")"
            if submit_web_notification(message, token):
                log("Web notification sent!")
            else:
                log("Error occurred while sending a web notification")
    else:
        log("Model metadata not found")


def submit_workflow_from_catalog(bucket_name, workflow_name, workflow_variables={}, token=""):
    result = False
    try:
        log("Connecting on " + proactive_url, token)
        gateway = proactive.ProActiveGateway(proactive_url, [])
        gateway.connect(
            username=user_credentials['ciLogin'],
            password=user_credentials['ciPasswd'])
        if gateway.isConnected():
            log("Connected to " + proactive_url, token)
            try:
                log("Submitting a workflow from the catalog", token)
                jobId = gateway.submitWorkflowFromCatalog(bucket_name, workflow_name, workflow_variables)
                assert jobId is not None
                assert isinstance(jobId, numbers.Number)
                workflow_path = bucket_name + "/" + workflow_name
                log("Workflow " + workflow_path + " submitted successfully with jobID: " + str(jobId), token)
                result = True
            finally:
                log("Disconnecting from " + proactive_url, token)
                gateway.disconnect()
                log("Disconnected from " + proactive_url, token)
                gateway.terminate()
                log("Connection finished from " + proactive_url, token)
        else:
            log("Couldn't connect to " + proactive_url, token)
    except Exception as e:
        log("Error while connecting on " + proactive_url, token)
        log(str(e), token)
    return result


def submit_web_notification(message, token):
    return submit_workflow_from_catalog("notification-tools", "Web_Notification", {'MESSAGE': message}, token)


def backup_previous_deployed_model():
    datetime_str = dt.today().strftime('%Y%m%d%H%M%S')
    if exists(CURRENT_MODEL_FILE) and isfile(CURRENT_MODEL_FILE):
        PREVIOUS_MODEL_FILE = join(UPLOAD_MODELS_FOLDER, 'model_' + datetime_str + MODEL_FILE_EXT)
        move(CURRENT_MODEL_FILE, PREVIOUS_MODEL_FILE)
        log("Current model file was moved to:\n" + PREVIOUS_MODEL_FILE)
    if exists(CURRENT_META_FILE) and isfile(CURRENT_META_FILE):
        PREVIOUS_META_FILE = join(UPLOAD_MODELS_FOLDER, 'model_' + datetime_str + META_FILE_EXT)
        move(CURRENT_META_FILE, PREVIOUS_META_FILE)
        log("Current model metadata file was moved to:\n" + PREVIOUS_META_FILE)


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
    user = connexion.request.form["user"]
    addr = connexion.request.remote_addr
    token = TOKENS.get(user)
    if not token:
        log('Invalid user: {u} ({a})'.format(u=user, a=addr))
        return "Invalid user"
    else:
        log('{u} token is {t} ({a})'.format(u=user, t=token, a=addr))
        return token


def predict_api(data: str) -> str:
    api_token = data['api_token']
    log("calling predict_api", api_token)
    if auth_token(api_token):
        if exists(CURRENT_MODEL_FILE) and isfile(CURRENT_MODEL_FILE):
            try:
                dataframe_json = data['dataframe_json']
                dataframe = pd.read_json(dataframe_json, orient='values')
                if DRIFT_ENABLED:
                    perform_drift_detection(dataframe, api_token)
                model = load(CURRENT_MODEL_FILE)
                log("model:\n" + str(model))
                log("dataframe:\n" + str(dataframe.head()))
                predictions = model.predict(dataframe.values)
                log("Model predictions done", api_token)
                return json.dumps(list(predictions))
            except Exception as e:
                return log(str(e), api_token)
        else:
            return log("Model file not found", api_token)
    else:
        return log("Invalid token", api_token)


def deploy_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling deploy_api", api_token)
    if auth_token(api_token):
        backup_previous_deployed_model()
        model_file = connexion.request.files['model_file']
        model_file.save(CURRENT_MODEL_FILE)
        log("The new model file was deployed successfully at:\n" + CURRENT_MODEL_FILE)
        # Check if model metadata exists and save it
        if "model_metadata_json" in connexion.request.form:
            log("Adding model metadata")
            model_metadata_json = connexion.request.form['model_metadata_json']
            model_metadata = pd.read_json(model_metadata_json, orient='values')
            # model_metadata = pd.read_json(model_metadata_json, orient='split')
            # print(model_metadata.head())
            model_metadata.to_pickle(CURRENT_META_FILE)
            log("The new model metadata file was saved successfully at:\n" + CURRENT_META_FILE)

        return log("Model deployed", api_token)
    else:
        return log("Invalid token", api_token)


def list_models_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling list_models_api", api_token)
    if auth_token(api_token):
        models_list = glob.glob(join(UPLOAD_MODELS_FOLDER, "*" + MODEL_FILE_EXT))
        log("List of deployed models:\n" + str(models_list))
        return json.dumps(models_list)
    else:
        return log("Invalid token", api_token)


def undeploy_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling undeploy_api", api_token)
    if auth_token(api_token):
        model_file = connexion.request.form["model_file"]
        if exists(model_file) and isfile(model_file):
            os.remove(model_file)
            # Check if the model has an associated metadata
            meta_file = model_file.replace(MODEL_FILE_EXT, META_FILE_EXT)
            if exists(meta_file) and isfile(meta_file):
                log("The model has an associated metadata")
                os.remove(meta_file)
        log("Model removed:\n" + str(model_file))
        return log("Model removed", api_token)
    else:
        return log("Invalid token", api_token)


def redeploy_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling redeploy_api", api_token)
    if auth_token(api_token):
        backup_previous_deployed_model()
        model_file = connexion.request.form["model_file"]
        if exists(model_file) and isfile(model_file):
            move(model_file, CURRENT_MODEL_FILE)
            # Check if the model has an associated metadata
            meta_file = model_file.replace(MODEL_FILE_EXT, META_FILE_EXT)
            if exists(meta_file) and isfile(meta_file):
                log("The model has an associated metadata")
                move(meta_file, CURRENT_META_FILE)
            # Done
            log("Model deployed successfully:\n" + str(model_file))
            log("From:\n" + str(model_file) + "\nTo:\n" + CURRENT_MODEL_FILE)
            return log("Model deployed", api_token)
        else:
            return log("Model file not found", api_token)
    else:
        return log("Invalid token", api_token)


def trace_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling trace_api", api_token)
    if auth_token(api_token):
        with open(TRACE_FILE) as f:
            lines = f.readlines()
        return lines
    else:
        return log("Invalid token", api_token)


def test_workflow_submission_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling test_workflow_submission_api", api_token)
    if auth_token(api_token):
        res = submit_workflow_from_catalog("basic-examples", "Print_File_Name", {'file': 'test_from_maas'}, api_token)
        return jsonify(res)
    else:
        return log("Invalid token", api_token)


def test_web_notification_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling test_web_notification_api", api_token)
    if auth_token(api_token):
        message = "MaaS notification test from " + get_token_user(api_token) + " (" + api_token + ")"
        res = submit_web_notification(message, api_token)
        return jsonify(res)
    else:
        return log("Invalid token", api_token)


# ----- Main entry point ----- #


if __name__ == '__main__':
    app = connexion.FlaskApp(__name__, port=9090, specification_dir=APP_BASE_DIR)
    CORS(app.app)
    app.add_api('ml_service_swagger.yaml', arguments={'title': 'Machine Learning Model Service'})
    
    if HTTPS_ENABLED:
        # from OpenSSL import SSL
        # context = SSL.Context(SSL.SSLv23_METHOD)
        # context.use_privatekey_file(join(APP_BASE_DIR, 'key_mas.pem'))  # yourserver.key
        # context.use_certificate_file(join(APP_BASE_DIR, 'certificate_mas.pem'))  # yourserver.crt
        context = (
            join(APP_BASE_DIR, 'certificate_mas.pem'),
            join(APP_BASE_DIR, 'key_mas.pem')
        )
        app.run(debug=DEBUG_ENABLED, ssl_context=context)
    else:
        app.run(debug=DEBUG_ENABLED)
