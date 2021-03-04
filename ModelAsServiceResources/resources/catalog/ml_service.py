#!/usr/bin/env python3

# import os, sys, bz2, uuid, pickle, json, connexion, wget
import numpy
import os
import sys
import glob
import json
import connexion
import subprocess
import numbers
import pandas as pd
import argparse

from cryptography.fernet import Fernet
from datetime import datetime as dt
from os.path import join, exists, isfile
from binascii import hexlify
from shutil import move
from flask_cors import CORS
from joblib import load
from flask import jsonify
from tempfile import TemporaryFile
from json import JSONEncoder

from scipy.stats import norm
from scipy.stats import wasserstein_distance
from urllib.parse import quote
from distutils.util import strtobool

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Install required Python libraries if they are not already installed
try:
    import proactive
except ImportError:
    install('proactive')
    import proactive
# try:
#     from json2html import *
# except ImportError:
#     install('json2html')
#     from json2html import *

try:
    from skmultiflow.drift_detection.hddm_w import HDDM_W
except ImportError:
    install('scikit-multiflow')
    from skmultiflow.drift_detection.hddm_w import HDDM_W

from skmultiflow.drift_detection import PageHinkley
from skmultiflow.drift_detection.adwin import ADWIN


# Environment variables
INSTANCE_PATH = os.getenv('INSTANCE_PATH') if os.getenv('INSTANCE_PATH') is not None else None

# General parameters
APP_BASE_DIR = ""
UPLOAD_MODELS_FOLDER = INSTANCE_PATH
MODEL_FILE_EXT = '.model'
# META_FILE_EXT = '.meta'
CURRENT_MODEL_FILE = join(UPLOAD_MODELS_FOLDER, 'model_last' + MODEL_FILE_EXT)  # default model path
# CURRENT_META_FILE = join(UPLOAD_MODELS_FOLDER, 'model_last' + META_FILE_EXT)    # default meta path
CURRENT_BASELINE_DATA = join(UPLOAD_MODELS_FOLDER, 'baseline_data' + '.csv') # baseline data path
TRACE_FILE = join(UPLOAD_MODELS_FOLDER, 'trace.txt')  # default trace file
CONFIG_FILE = join(UPLOAD_MODELS_FOLDER, 'config.json')  # default config file
PREDICTIONS_FILE = join(UPLOAD_MODELS_FOLDER, 'predictions.csv')  # default predictions file
TOKENS = { # user api tokens
    'user': hexlify(os.urandom(16)).decode(),  # api key
    'test': hexlify(os.urandom(16)).decode()
}

DEBUG_ENABLED = True if (os.getenv('DEBUG_ENABLED') is not None and os.getenv('DEBUG_ENABLED').lower() == "true") else False
TRACE_ENABLED = True if (os.getenv('TRACE_ENABLED') is not None and os.getenv('TRACE_ENABLED').lower() == "true") else False
DRIFT_ENABLED = True if (os.getenv('DRIFT_ENABLED') is not None and os.getenv('DRIFT_ENABLED').lower() == "true") else False
# DRIFT_THRESHOLD = float(os.getenv('DRIFT_THRESHOLD')) if os.getenv('DRIFT_THRESHOLD') is not None else None
DRIFT_NOTIFICATION = True if (os.getenv('DRIFT_NOTIFICATION') is not None and os.getenv('DRIFT_NOTIFICATION').lower() == "true") else False
LOG_PREDICTIONS = True if (os.getenv('LOG_PREDICTIONS') is not None and os.getenv('LOG_PREDICTIONS').lower() == "true") else False
HTTPS_ENABLED = True if (os.getenv('HTTPS_ENABLED') is not None and os.getenv('HTTPS_ENABLED').lower() == "true") else False
USER_KEY = os.getenv('USER_KEY')
assert USER_KEY is not None, "USER_KEY is required!"
USER_KEY = str(USER_KEY).encode()


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


# Check if there is already a configuration file
if not isfile(CONFIG_FILE):
    print("Generating the configuration file")
    config = {
        'DEBUG_ENABLED': DEBUG_ENABLED,
        'TRACE_ENABLED': TRACE_ENABLED,
        'DRIFT_ENABLED': DRIFT_ENABLED,
        # 'DRIFT_THRESHOLD': DRIFT_THRESHOLD,
        'DRIFT_NOTIFICATION': DRIFT_NOTIFICATION,
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
    
def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__


def perform_drift_detection(predict_dataframe, dataframe, feature_names, detector, token="") -> str :
    log("calling perform_drift_detection", token)
    log("data drift detection method: " + detector)
    baseline_data = dataframe.values.tolist()
    predict_data = predict_dataframe.values.tolist()
    overall_data = list()
    for a in baseline_data:
        overall_data.append(a)
    for b in predict_data:
        overall_data.append(b)
    overall_dataframe = pd.DataFrame(overall_data, columns=feature_names)
    drifts = dict()
    window = len(baseline_data)
    for feature in feature_names:
        detected_drifts_indices = list()
        # HDDM
        if detector == "HDDM":
            hddm_w = HDDM_W()
            for i in range(len(overall_dataframe[feature])):
                hddm_w.add_element(float(overall_dataframe[feature][i]))
                if hddm_w.detected_change() and i >= window:
                    detected_drifts_indices.append(i-window)
        # Page Hinkley
        if detector == "Page Hinkley":
            ph = PageHinkley()
            for i in range(len(overall_dataframe[feature])):
                ph.add_element(float(overall_dataframe[feature][i]))
                if ph.detected_change() and i >= window:
                    detected_drifts_indices.append(i-window)
        # ADWIN
        if detector == "ADWIN":
            adwin = ADWIN()
            for i in range(len(overall_dataframe[feature])):
                adwin.add_element(float(overall_dataframe[feature][i]))
                if adwin.detected_change() and i >= window:
                    detected_drifts_indices.append(i-window)
        # Check for detected drifts
        if len(detected_drifts_indices) != 0:
            log("Data drift detected in feature: " + feature)
            log("The drifted rows are: " + str(detected_drifts_indices))
            drifts[feature] = detected_drifts_indices
            if get_config('DRIFT_NOTIFICATION'):
                log("Sending a web notification", token)
                message = "MaaS data drift detected from " + get_token_user(token) + " (" + token + ")"
                if submit_web_notification(message, token):
                    log("Web notification sent!")
                else:
                    log("Error occurred while sending a web notification")
    return json.dumps(drifts)
    
    # This code is commented in case needed in future for drift detection
    # log("calling perform_drift_detection", token)
    # if exists(CURRENT_META_FILE) and isfile(CURRENT_META_FILE):
    #     # log("The model has an associated metadata")
    #     model_metadata = pd.read_pickle(CURRENT_META_FILE)
    #     log("model_metadata:\n" + str(model_metadata), token)
    #     # log("Calculating data drift measures", token)
    #     predict_mean = predict_dataframe.mean(axis=0)  # mean
    #     predict_std = predict_dataframe.std(axis=0)    # standard deviation
    #     # predict_metadata = pd.DataFrame({'Mean': predict_mean, 'Std': predict_std}).T
    #     predict_metadata = pd.DataFrame({0: predict_mean, 1: predict_std}).T
    #     log("predict_metadata:\n" + str(predict_metadata), token)
    #     size_data = len(model_metadata.columns)
    #     model_metadata_normal = norm.rvs(
    #         size=size_data,
    #         loc=model_metadata.iloc[0],    # mean
    #         scale=model_metadata.iloc[1])  # std
    #     predict_metadata_normal = norm.rvs(
    #         size=size_data,
    #         loc=predict_metadata.iloc[0],    # mean
    #         scale=predict_metadata.iloc[1])  # std
    #     # Wasserstein distance
    #     score = wasserstein_distance(model_metadata_normal, predict_metadata_normal)
    #     log("Wasserstein distance: " + str(score), token)
    #     # Data drift detection
    #     DRIFT_THRESHOLD = get_config('DRIFT_THRESHOLD')
    #     log("Drift threshold was set to: " + str(DRIFT_THRESHOLD), token)
    #     # Send web notification alerts
    #     if DRIFT_THRESHOLD is not None and score > DRIFT_THRESHOLD:
    #         log("Data drift detected!", token)
    #         if get_config('DRIFT_NOTIFICATION'):
    #             log("Sending a web notification", token)
    #             message = "MaaS data drift detected from " + get_token_user(token) + " (" + token + ")"
    #             if submit_web_notification(message, token):
    #                 log("Web notification sent!")
    #             else:
    #                 log("Error occurred while sending a web notification")
    # else:
    #     log("Model metadata not found")


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
    # if exists(CURRENT_META_FILE) and isfile(CURRENT_META_FILE):
    #     PREVIOUS_META_FILE = join(UPLOAD_MODELS_FOLDER, 'model_' + datetime_str + META_FILE_EXT)
    #     move(CURRENT_META_FILE, PREVIOUS_META_FILE)
    #     log("Current model metadata file was moved to:\n" + PREVIOUS_META_FILE)


def auth_token(token):
    for _, key in TOKENS.items():
        if key == token:
            return True
    return False


def get_token_user(token):
    for user, key in TOKENS.items():
        if key == token:
            return user
    return None


def color_drift_detection(val):
    color = 'red' if ("drift detected" in val) else 'black'
    return 'color: %s' % color


def highlight_drift_detection(values):
    return ['background-color: yellow' if ("drift detected" in v) else '' for v in values]


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
    predict_drifts = dict()
    drifts_json = "No baseline data added for the drift detection."
    api_token = data['api_token']
    log("calling predict_api", api_token)
    detector = data['detector']
    dataframe = pd.DataFrame()
    feature_names = list()
    if exists(CURRENT_BASELINE_DATA): # and isfile(CURRENT_BASELINE_DATA):
        dataframe = pd.read_csv(CURRENT_BASELINE_DATA)
    if not dataframe.empty:
        feature_names = dataframe.columns
    if auth_token(api_token):
        if exists(CURRENT_MODEL_FILE) and isfile(CURRENT_MODEL_FILE):
            try:
                # dataframe_json = data['dataframe_json']
                # dataframe = pd.read_json(dataframe_json, orient='values')
                predict_dataframe_json = data['predict_dataframe_json']
                predict_dataframe = pd.read_json(predict_dataframe_json, orient='values')
                if get_config('DRIFT_ENABLED') and dataframe.empty == False:
                    drifts_json = perform_drift_detection(predict_dataframe, dataframe, feature_names, detector, api_token)
                else:
                    drifts_json = "Drift detection is not enabled."
                model = load(CURRENT_MODEL_FILE)
                log("model:\n" + str(model))
                log("dataframe:\n" + str(dataframe.head()))
                predictions = model.predict(predict_dataframe.values)
                if get_config('LOG_PREDICTIONS'):
                    log("Logging predictions")
                    dataframe_predictions = predict_dataframe.assign(predictions=predictions)
                    dataframe_predictions.to_csv(PREDICTIONS_FILE, header=False, index=False, mode="a")
                    log("Done")
                log("Model predictions done", api_token)
                predict_drifts['predictions'] = list(predictions)
                predict_drifts['drifts'] = drifts_json
                return json.dumps(predict_drifts) 
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
        if "baseline_data" in connexion.request.files:
            baseline_data = connexion.request.files['baseline_data']
            baseline_data.save(CURRENT_BASELINE_DATA)
            # log("The new baseline data file was deployed successfully at:\n" + CURRENT_BASELINE_DATA)
        # Check if model metadata exists and save it
        # if "model_metadata_json" in connexion.request.form:
        #     log("Adding model metadata")
        #     model_metadata_json = connexion.request.form['model_metadata_json']
        #     log("model_metadata_json:\n" + str(model_metadata_json), api_token)
        #     model_metadata = pd.read_json(model_metadata_json, orient='values')
        #     # model_metadata = pd.read_json(model_metadata_json, orient='split')
        #     # print(model_metadata.head())
        #     log("model_metadata:\n" + str(model_metadata), api_token)
        #     model_metadata.to_pickle(CURRENT_META_FILE)
        #     log("The new model metadata file was saved successfully at:\n" + CURRENT_META_FILE)
        #
        # Check if debug is enabled
        if "debug_enabled" in connexion.request.form:
            debug_enabled = connexion.request.form['debug_enabled']
            log("Updating DEBUG_ENABLED to " + debug_enabled)
            set_config('DEBUG_ENABLED', bool(strtobool(debug_enabled)))
        # Check if trace is enabled
        if "trace_enabled" in connexion.request.form:
            trace_enabled = connexion.request.form['trace_enabled']
            log("Updating TRACE_ENABLED to " + trace_enabled)
            set_config('TRACE_ENABLED', bool(strtobool(trace_enabled)))
        # Check if drift is enabled
        if "drift_enabled" in connexion.request.form:
            drift_enabled = connexion.request.form['drift_enabled']
            log("Updating DRIFT_ENABLED to " + drift_enabled)
            set_config('DRIFT_ENABLED', bool(strtobool(drift_enabled)))
        # Check if the drift threshold is set
        # if "drift_threshold" in connexion.request.form:
        #     drift_threshold = connexion.request.form['drift_threshold']
        #     log("Updating DRIFT_THRESHOLD to " + drift_threshold)
        #     set_config('DRIFT_THRESHOLD', float(drift_threshold))
        # Check if the drift notification is enabled
        if "drift_notification" in connexion.request.form:
            drift_notification = connexion.request.form['drift_notification']
            log("Updating DRIFT_NOTIFICATION to " + drift_notification)
            set_config('DRIFT_NOTIFICATION', bool(strtobool(drift_notification)))
        # Check if the log predictions is enabled
        if "log_predictions" in connexion.request.form:
            log_predictions = connexion.request.form['log_predictions']
            log("Updating LOG_PREDICTIONS to " + log_predictions)
            set_config('LOG_PREDICTIONS', bool(strtobool(log_predictions)))
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
        # if exists(model_file) and isfile(model_file):
        #     os.remove(model_file)
        #     # Check if the model has an associated metadata
        #     meta_file = model_file.replace(MODEL_FILE_EXT, META_FILE_EXT)
        #     if exists(meta_file) and isfile(meta_file):
        #         log("The model has an associated metadata")
        #         os.remove(meta_file)
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
            # meta_file = model_file.replace(MODEL_FILE_EXT, META_FILE_EXT)
            # if exists(meta_file) and isfile(meta_file):
            #     log("The model has an associated metadata")
            #     move(meta_file, CURRENT_META_FILE)
            # Done
            log("Model deployed successfully:\n" + str(model_file))
            log("From:\n" + str(model_file) + "\nTo:\n" + CURRENT_MODEL_FILE)
            return log("Model deployed", api_token)
        else:
            return log("Model file not found", api_token)
    else:
        return log("Invalid token", api_token)


def update_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling update_api", api_token)
    if auth_token(api_token):
        # Update the debug parameter
        debug_enabled = connexion.request.form['debug_enabled']
        log("Updating DEBUG_ENABLED to " + debug_enabled)
        set_config('DEBUG_ENABLED', bool(strtobool(debug_enabled)))
        # Update the trace parameter
        trace_enabled = connexion.request.form['trace_enabled']
        log("Updating TRACE_ENABLED to " + trace_enabled)
        set_config('TRACE_ENABLED', bool(strtobool(trace_enabled)))
        # Update the drift parameter
        drift_enabled = connexion.request.form['drift_enabled']
        log("Updating DRIFT_ENABLED to " + drift_enabled)
        set_config('DRIFT_ENABLED', bool(strtobool(drift_enabled)))
        # Update the drift threshold parameter
        # drift_threshold = connexion.request.form['drift_threshold']
        # log("Updating DRIFT_THRESHOLD to " + drift_threshold)
        # set_config('DRIFT_THRESHOLD', float(drift_threshold))
        # Update the drift notification parameter
        drift_notification = connexion.request.form['drift_notification']
        log("Updating DRIFT_NOTIFICATION to " + drift_notification)
        set_config('DRIFT_NOTIFICATION', bool(strtobool(drift_notification)))
        # Update the log predictions parameter
        log_predictions = connexion.request.form['log_predictions']
        log("Updating LOG_PREDICTIONS to " + log_predictions)
        set_config('LOG_PREDICTIONS', bool(strtobool(log_predictions)))
        if "baseline_data" in connexion.request.files:
            #log("Updating baseline data")
            baseline_data_updated = connexion.request.files['baseline_data']
            baseline_data_updated.save(CURRENT_BASELINE_DATA)
        return log("Service parameters updated", api_token)
    else:
        return log("Invalid token", api_token)


def trace_preview_api(key) -> str:
    if USER_KEY == key.encode():
        if exists(TRACE_FILE) and isfile(TRACE_FILE):
            header = ["Date Time", "Token", "Traceability information"]
            result = ""
            with open(TRACE_FILE) as f, TemporaryFile("w+") as t:
                for line in f:
                    ln = len(line.strip().split("|"))
                    if ln < 3:
                        line = "||" + line
                    t.write(line)
                t.seek(0)
                trace_dataframe = pd.read_csv(t, sep='|', names=header, engine='python')
                trace_dataframe.fillna('', inplace=True)
                # result = (trace_dataframe.style.hide_index()
                #          .applymap(color_drift_detection)
                #          .apply(highlight_drift_detection)
                #          .set_properties(subset=['Date Time'], **{'width': '150px'})
                #          .set_properties(subset=['Token'], **{'width': '250px'})
                #          .render(table_title="Audit & Traceability"))
            # Add config information
            with open(CONFIG_FILE) as f:
                config = json.load(f)
            # result = json2html.convert(json=config) + result
            dataframe_config = pd.DataFrame.from_records([config])
            # result = dataframe_config.style.hide_index().render() + "<br/>" + result
            # .set_table_styles([{'selector': '', 'props': [('border', '4px solid #7a7')]}])
            config_result = dataframe_config.to_html(escape=False, classes='table table-bordered', justify='center', index=False)
            trace_result = trace_dataframe.to_html(escape=False, classes='table table-bordered table-striped', justify='center', index=False)
            css_style = """
            div {
            weight: 100%;
            }
            """
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
                result = "<p><a href='predictions_preview?key="+quote(key)+"' target='_blank'>Click here to visualize the predictions</a></p>" + result
            return result
        else:
            return log("Trace file is empty", key)
    else:
        return log("Invalid key", key)


# def trace_api() -> str:
#     api_token = connexion.request.form["api_token"]
#     log("calling trace_api", api_token)
#     if auth_token(api_token):
#         with open(TRACE_FILE) as f:
#             lines = f.readlines()
#         return lines
#     else:
#         return log("Invalid token", api_token)


def predictions_preview_api(key) -> str:
    if USER_KEY == key.encode():
        if exists(PREDICTIONS_FILE) and isfile(PREDICTIONS_FILE):
            predictions_dataframe = pd.read_csv(PREDICTIONS_FILE, header=None)
            predictions_dataframe.columns = [*predictions_dataframe.columns[:-1], 'predictions']
            predictions_dataframe.fillna('', inplace=True)
            # result = (predictions_dataframe.style.hide_index()
            #          .render(table_title="Predictions"))
            result = predictions_dataframe.to_html(escape=False, classes='table table-bordered table-striped', justify='center', index=False)
            result = """
            <!DOCTYPE html>
            <html>
              <head>
                <meta charset="UTF-8">
                  <title>Machine Learning Preview</title>
                  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
              </head>
                <body class="container">
                  <h1 class="text-center my-4" style="color:#003050;">Predictions Preview</h1>
                  <div style="text-align:center">{0}</div>
                </body></html>""".format(result)
            return result
        else:
            return log("Predictions are empty", key)
    else:
        return log("Invalid key", key)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9090, help="set the port that will be used to deploy the service")
    args = parser.parse_args()
    app = connexion.FlaskApp(__name__, port=args.port, specification_dir=APP_BASE_DIR)
    CORS(app.app)
    app.add_api('ml_service-api.yaml', arguments={'title': 'Machine Learning Model Service'})
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
