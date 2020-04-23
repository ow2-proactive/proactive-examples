#!/usr/bin/env python3

# import os, sys, bz2, uuid, pickle, json, connexion, wget
# import numpy as np
import os
import glob
import json
import connexion
import pandas as pd

from datetime import datetime as dt
from os.path import join, exists, isfile
from binascii import hexlify
from shutil import move
from flask_cors import CORS
from joblib import load

# General parameters
APP_BASE_DIR = "/model_as_a_service"
UPLOAD_MODELS_FOLDER = "/model_as_a_service"  # model folder
CURRENT_MODEL_FILE = join(UPLOAD_MODELS_FOLDER, 'model_last.pkl')  # default model path
TRACE_FILE = join(UPLOAD_MODELS_FOLDER, 'trace.txt')  # default trace file
API_KEY = hexlify(os.urandom(16)).decode()  # api key
TOKENS = {'user': API_KEY}  # user tokens

# Environment variables
DEBUG_ENABLED = True if (os.getenv('DEBUG_ENABLED') is not None and os.getenv('DEBUG_ENABLED').lower() == "true") else False
TRACE_ENABLED = True if (os.getenv('TRACE_ENABLED') is not None and os.getenv('TRACE_ENABLED').lower() == "true") else False
HTTPS_ENABLED = True if (os.getenv('HTTPS_ENABLED') is not None and os.getenv('HTTPS_ENABLED').lower() == "true") else False


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


def backup_previous_deployed_model():
    if exists(CURRENT_MODEL_FILE) and isfile(CURRENT_MODEL_FILE):
        datetime_str = dt.today().strftime('%Y%m%d%H%M%S')
        PREVIOUS_MODEL_FILE = join(UPLOAD_MODELS_FOLDER, 'model_' + datetime_str + '.pkl')
        move(CURRENT_MODEL_FILE, PREVIOUS_MODEL_FILE)
        log("Current model file was moved to:\n" + PREVIOUS_MODEL_FILE)


def get_token_api(user) -> str:
    user = connexion.request.form["user"]
    token = TOKENS.get(user)
    if not token:
        log('Invalid user: {u}'.format(u=user))
        return "Invalid user"
    else:
        log('{u} token is {t}'.format(u=user, t=token))
        return token


def predict_api(data: str) -> str:
    api_token = data['api_token']
    log("calling predict_api", api_token)
    if api_token == API_KEY:
        if exists(CURRENT_MODEL_FILE) and isfile(CURRENT_MODEL_FILE):
            try:
                dataframe_json = data['dataframe_json']
                dataframe = pd.read_json(dataframe_json, orient='values')
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
    if api_token == API_KEY:
        backup_previous_deployed_model()
        model_file = connexion.request.files['model_file']
        model_file.save(CURRENT_MODEL_FILE)
        log("The new model file was deployed successfully at:\n" + CURRENT_MODEL_FILE)
        return log("Model deployed", api_token)
    else:
        return log("Invalid token", api_token)


def list_models_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling list_models_api", api_token)
    if api_token == API_KEY:
        models_list = glob.glob(join(UPLOAD_MODELS_FOLDER, "*.pkl"))
        log("List of deployed models:\n" + str(models_list))
        return json.dumps(models_list)
    else:
        return log("Invalid token", api_token)


def undeploy_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling undeploy_api", api_token)
    if api_token == API_KEY:
        model_file = connexion.request.form["model_file"]
        if exists(model_file) and isfile(model_file):
            os.remove(model_file)
        log("Model removed:\n" + str(model_file))
        return log("Model removed", api_token)
    else:
        return log("Invalid token", api_token)


def redeploy_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling redeploy_api", api_token)
    if api_token == API_KEY:
        backup_previous_deployed_model()
        model_file = connexion.request.form["model_file"]
        if exists(model_file) and isfile(model_file):
            move(model_file, CURRENT_MODEL_FILE)
            log("Model deployed successfully:\n" + str(model_file))
            return log("Model deployed", api_token)
        else:
            return log("Model file not found", api_token)
    else:
        return log("Invalid token", api_token)


def trace_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling trace_api", api_token)
    if api_token == API_KEY:
        with open(TRACE_FILE) as f:
            lines = f.readlines()
        return lines
    else:
        return log("Invalid token", api_token)


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
