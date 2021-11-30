# Copyright Activeeon 2007-2021. All rights reserved.
#!/usr/bin/env python3

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
import shutil
import dash_utils

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
from flask import render_template
from os import environ, path

from urllib.parse import quote
from distutils.util import strtobool


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Install required Python libraries if they are not already installed
try:
    import proactive
except ImportError:
    install('proactive')
    import proactive

# DDD libraries
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
CURRENT_BASELINE_DATA = join(UPLOAD_MODELS_FOLDER, 'baseline_data' + '.csv')  # baseline data path
TRACE_FILE = join(UPLOAD_MODELS_FOLDER, 'trace.txt')  # default trace file
CONFIG_FILE = join(UPLOAD_MODELS_FOLDER, 'config.json')  # default config file
TOKENS = {  # user api tokens
    'user': hexlify(os.urandom(16)).decode(),  # api key
    'test': hexlify(os.urandom(16)).decode()
}

DEBUG_ENABLED = True if (os.getenv('DEBUG_ENABLED') is not None and os.getenv('DEBUG_ENABLED').lower() == "true") else False
TRACE_ENABLED = True if (os.getenv('TRACE_ENABLED') is not None and os.getenv('TRACE_ENABLED').lower() == "true") else False
GPU_ENABLED = True if (os.getenv('GPU_ENABLED') is not None and os.getenv('GPU_ENABLED').lower() == "true") else False
HTTPS_ENABLED = True if (os.getenv('HTTPS_ENABLED') is not None and os.getenv('HTTPS_ENABLED').lower() == "true") else False
USER_KEY = os.getenv('USER_KEY')
assert USER_KEY is not None, "USER_KEY is required!"
USER_KEY = str(USER_KEY).encode()
os.environ['MODELS_PATH'] = join(INSTANCE_PATH, "models")

# Required imports for Nvidia Rapids Usage
if GPU_ENABLED:
    import cudf
    from cudf import read_json, read_csv

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
        'GPU_ENABLED': GPU_ENABLED,
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

def perform_drift_detection(predict_dataframe, dataframe, feature_names, detector, drift_notification, token="") -> str:
    log("[INFO] Calling perform_drift_detection", token)
    log("[INFO] Selected data drift detection method: " + detector)
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
                    detected_drifts_indices.append(i - window)
        # Page Hinkley
        if detector == "Page Hinkley":
            ph = PageHinkley()
            for i in range(len(overall_dataframe[feature])):
                ph.add_element(float(overall_dataframe[feature][i]))
                if ph.detected_change() and i >= window:
                    detected_drifts_indices.append(i - window)
        # ADWIN
        if detector == "ADWIN":
            adwin = ADWIN()
            for i in range(len(overall_dataframe[feature])):
                adwin.add_element(float(overall_dataframe[feature][i]))
                if adwin.detected_change() and i >= window:
                    detected_drifts_indices.append(i - window)
        # Check for detected drifts
        if len(detected_drifts_indices) != 0:
            log("[INFO] Data drift detected in feature: " + feature)
            log("[INFO] The drifted rows are: " + str(detected_drifts_indices))
            drifts[feature] = detected_drifts_indices
            if drift_notification:
                log("[INFO] Sending a web notification", token)
                message = "MaaS data drift detected from " + get_token_user(token) + " (" + token + ")"
                if submit_web_notification(message, token):
                    log("[INFO] Web notification sent!")
                else:
                    log("[ERROR] Error occurred while sending a web notification")
    return json.dumps(drifts, cls=NpEncoder)


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
            log("[ERROR] Couldn't connect to " + proactive_url, token)
    except Exception as e:
        log("[ERROR] Error while connecting on " + proactive_url, token)
        log(str(e), token)
    return result


def submit_web_notification(message, token):
    return submit_workflow_from_catalog("notification-tools", "Web_Notification", {'MESSAGE': message}, token)


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
    addr = connexion.request.remote_addr
    token = TOKENS.get(user)
    if not token:
        log('[ERROR] Invalid user: {u} ({a})'.format(u=user, a=addr))
        return "Invalid user"
    else:
        log('[INFO] {u} token is {t} ({a})'.format(u=user, t=token, a=addr))
        return token


def predict_api(data: str) -> str:
    api_token = data['api_token']
    if auth_token(api_token):
        predict_drifts = dict()
        drifts_json = "No baseline data added for the drift detection."
        log("[INFO] Calling predict_api", api_token)
        detector = data['detector']
        feature_names = list()
        model_name = data["model_name"]
        model_version = data["model_version"]
        drift_enabled = data["drift_enabled"]
        drift_notification = data["drift_notification"]
        save_predictions = data["save_predictions"]
        model_file_name = "model_" + str(model_version) + ".model"
        baseline_data_file_name = "baseline_data_" + str(model_version) + ".csv"
        predictions_data_file_name = "predictions_data_" + str(model_version) + ".csv"
        MODELS_PATH = os.environ['MODELS_PATH']
        model_file_path = join(MODELS_PATH, str(model_name), str(model_version), model_file_name)
        baseline_data_file_path = join(MODELS_PATH, str(model_name), str(model_version), baseline_data_file_name)
        predictions_csv_file = join(MODELS_PATH, str(model_name), str(model_version), predictions_data_file_name)

        if GPU_ENABLED:
            dataframe = cudf.DataFrame()
        else:
            dataframe = pd.DataFrame()

        if exists(baseline_data_file_path) and drift_enabled:  # and isfile(CURRENT_BASELINE_DATA):
            if GPU_ENABLED:
                dataframe = read_csv(baseline_data_file_path)
            else:
                log("[INFO] baseline_data_file_path:\n" + str(baseline_data_file_path))
                dataframe = pd.read_csv(baseline_data_file_path)
        if not dataframe.empty:
            feature_names = dataframe.columns
        if exists(model_file_path) and isfile(model_file_path):
            try:
                predict_dataframe_json = data['predict_dataframe_json']
                predict_dataframe = pd.read_json(predict_dataframe_json, orient='values')
                if GPU_ENABLED:
                    predict_dataframe = cudf.DataFrame.from_pandas(predict_dataframe)
                if drift_enabled and dataframe.empty == False:
                    drifts_json = perform_drift_detection(predict_dataframe, dataframe, feature_names, detector, drift_notification, api_token)
                else:
                    drifts_json = "Drift detection is not enabled."
                log("[INFO] model_file_path:\n" + str(model_file_path))
                model = load(model_file_path)

                predictions = model.predict(predict_dataframe)
                if save_predictions:
                    log("[INFO] Saving predictions")
                    dataframe_predictions = predict_dataframe.assign(predictions=predictions)
                    dataframe_predictions = dataframe_predictions.assign(api_token=api_token)
                    dataframe_predictions.to_csv(predictions_csv_file, mode='a', header=False, index=False)
                log("[INFO] Precitions provided", api_token)
                if GPU_ENABLED:
                    predict_drifts['predictions'] = list(predictions.to_pandas())
                else:
                    predict_drifts['predictions'] = list(predictions)
                predict_drifts['drifts'] = drifts_json
                return json.dumps(predict_drifts, cls=NpEncoder)
            except Exception as e:
                return log(str(e), api_token)
        else:
            return log("[ERROR] Model file not found", api_token)
    else:
        return log("[ERROR] Invalid token", api_token)


def deploy_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("[INFO] Calling deploy_api", api_token)
    if auth_token(api_token):
        model_file = connexion.request.files['model_file']
        model_name = connexion.request.form["model_name"]
        model_download_path = os.environ['MODELS_PATH'] + "/" + model_name
        download_model = True
        # if model_version is empty, model_version will be set on None
        try:
            model_version = int(connexion.request.form["model_version"])
        except:
            model_version = None
            pass
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
            deployment_status = "[INFO] The version " + str(model_version) + " of the model " + str(
                model_name) + " was successfully deployed."
        else:
            version_path = model_download_path + "/" + str(model_version)
            deployment_status = "[INFO] The version " + str(model_version) + " of the model " + str(
                model_name) + " was successfully deployed."
            if os.path.isdir(version_path):
                download_model = False
                log("[WARN] This model version already exists. \The uploaded model version will be ignored. The existing version will be deployed.",api_token)
                deployment_status = "[INFO] The model version " + str(model_version) + " of the model " + str(model_name) + " is already deployed. The uploaded model version will be ignored."

        # Model Downloading
        # if the specified model version doesn't exist in the directory,
        # the zip file uploaded by the user will be downloaded
        if download_model:
            # datetime_str = dt.today().strftime('%Y%m%d%H%M%S')
            model_file_name = "model_" + str(model_version)
            version_path = model_download_path + "/" + str(model_version)
            os.makedirs(version_path)
            model_file_path = version_path + "/" + model_file_name + ".model"
            log("[INFO] Downloading the new model in " + str(version_path), api_token)
            model_file.save(model_file_path)
        if "baseline_data" in connexion.request.files:
            baseline_data = connexion.request.files['baseline_data']
            baseline_data_file_name = "baseline_data_" + str(model_version)
            baseline_data_file_path = version_path + "/" + baseline_data_file_name + ".csv"
            baseline_data.save(baseline_data_file_path)
        # Check if debug is enabled
        if "debug_enabled" in connexion.request.form:
            debug_enabled = connexion.request.form['debug_enabled']
            log("[INFO] Updating DEBUG_ENABLED to " + debug_enabled)
            set_config('DEBUG_ENABLED', bool(strtobool(debug_enabled)))
        # Check if trace is enabled
        if "trace_enabled" in connexion.request.form:
            trace_enabled = connexion.request.form['trace_enabled']
            log("[INFO] Updating TRACE_ENABLED to " + trace_enabled)
            set_config('TRACE_ENABLED', bool(strtobool(trace_enabled)))
        return log(deployment_status, api_token)
    else:
        return log("[ERROR] Invalid token", api_token)


def list_saved_models_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("[INFO] Calling list_saved_models_api", api_token)
    if auth_token(api_token):
        json_format = connexion.request.form["json_format"]
        if json_format == "true":
            model_download_path = 'tree -J ' + os.environ['MODELS_PATH']
        else:
            model_download_path = 'tree ' + os.environ['MODELS_PATH']
        tree_model_download_path = os.popen(model_download_path).read()
        return tree_model_download_path
    else:
        return log("[ERROR] Invalid token", api_token)


def delete_deployed_model_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("[INFO] Calling delete_deployed_model_api", api_token)
    if auth_token(api_token):
        model_name = connexion.request.form["model_name"]
        model_download_path = os.environ['MODELS_PATH'] + "/" + model_name
        try:
            model_version = int(connexion.request.form["model_version"])
        except:
            model_version = None
            pass
        if not os.path.exists(model_download_path):
            clean_status = "[ERROR] The model folder " + str(model_download_path) + " doesn't exist."
        elif model_version is None:
            if os.path.exists(model_download_path):
                shutil.rmtree(model_download_path)
                clean_status = "[INFO] The model folder " + str(model_download_path) + " was successfully deleted."
        else:
            model_version_path = model_download_path + "/" + str(model_version)
            if os.path.exists(model_version_path):
                shutil.rmtree(model_version_path)
                clean_status = "[INFO] The model version folder " + str(
                    model_version_path) + " was successfully deleted."
            else:
                clean_status = "[ERROR] The model version folder " + str(model_version_path) + " doesn't exist."
        return log(clean_status, api_token)
    else:
        return log("[ERROR] Invalid token", api_token)


def update_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling update_api", api_token)
    if auth_token(api_token):
        # Update the debug parameter
        debug_enabled = connexion.request.form['debug_enabled']
        log("[INFO] Updating DEBUG_ENABLED to " + debug_enabled)
        set_config('DEBUG_ENABLED', bool(strtobool(debug_enabled)))
        # Update the trace parameter
        trace_enabled = connexion.request.form['trace_enabled']
        log("[INFO] Updating TRACE_ENABLED to " + trace_enabled)
        set_config('TRACE_ENABLED', bool(strtobool(trace_enabled)))
        return log("[INFO] Service parameters updated", api_token)
    else:
        return log("[ERROR] Invalid token", api_token)

def get_predictions_api(api_token,model_name,model_version):
    addr = connexion.request.remote_addr
    predictions_file_path = os.environ['MODELS_PATH'] + "/" + model_name + "/" + str(model_version) + "/predictions_data_" + str(model_version) + ".csv"
    if auth_token(api_token):
        if os.path.exists(predictions_file_path):
            predictions_df = pd.read_csv(predictions_file_path, delimiter=',')
            predictions_df.loc[predictions_df[predictions_df.columns[-1]] == api_token]
            predictions_df = predictions_df.iloc[:, :-1]
            predictions_json = predictions_df.to_json(orient='split')
            return predictions_json
        else:
            get_prediction_status = "[WARN] No predictions saved for the version " + str(model_version) + " of model " + model_name + " using token " + str(api_token)
            return log(get_prediction_status, api_token)
        return log("[ERROR] Invalid token", api_token)


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
            # Add config information
            with open(CONFIG_FILE) as f:
                config = json.load(f)
            dataframe_config = pd.DataFrame.from_records([config])
            config_result = dataframe_config.to_html(escape=False, classes='table table-bordered', justify='center',index=False)
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
            if (os.path.isdir(os.environ['MODELS_PATH'])):
                directory_contents = os.listdir(os.environ['MODELS_PATH'])
                if (len(directory_contents) > 0):
                    result = "<p><a href='maas_analytics?key=" + quote(key) + "' target='_blank'>Click here for MaaS_ML data analytics</a></p>" + result
            return result
        else:
            return log("[WARN] Trace file is empty", key)
    else:
        return log("[ERROR] Invalid key", key)


def maas_analytics_api(key) -> str:
    if USER_KEY == key.encode():
        return render_template(
            'index.jinja2',
            title='Plotly Dash Flask Tutorial',
            description='Embed Plotly Dash into your Flask applications.',
            template='home-template',
            body="This is a homepage served with Flask."
        )
    else:
        return log("Invalid key", key)


def test_workflow_submission_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling test_workflow_submission_api", api_token)
    if auth_token(api_token):
        res = submit_workflow_from_catalog("basic-examples", "Print_File_Name", {'file': 'test_from_maas'}, api_token)
        return jsonify(res)
    else:
        return log("[ERROR] Invalid token", api_token)


def test_web_notification_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling test_web_notification_api", api_token)
    if auth_token(api_token):
        message = "MaaS notification test from " + get_token_user(api_token) + " (" + api_token + ")"
        res = submit_web_notification(message, api_token)
        return jsonify(res)
    else:
        return log("[ERROR] Invalid token", api_token)



# ----- Main entry point ----- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9090, help="set the port that will be used to deploy the service")
    args = parser.parse_args()
    app = connexion.FlaskApp(__name__, port=args.port, specification_dir=APP_BASE_DIR)
    CORS(app.app)
    app.add_api('ml_service-api.yaml', arguments={'title': 'Machine Learning Model Service'})
    dash_utils.init_dashboard(app.app)
    if HTTPS_ENABLED:
        context = (
            join(APP_BASE_DIR, 'certificate_mas.pem'),
            join(APP_BASE_DIR, 'key_mas.pem')
        )
        app.run(debug=DEBUG_ENABLED, ssl_context=context)
    else:
        app.run(debug=DEBUG_ENABLED)