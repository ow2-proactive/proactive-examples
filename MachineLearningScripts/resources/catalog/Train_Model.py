# -*- coding: utf-8 -*-
"""Proactive Train Model for Machine Learning

This module contains the Python script for the Train Model task.
"""
import ssl
import urllib.request
import bz2
import json
import pickle
import sys
import uuid
import shap

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

global variables, resultMetadata, resultMap

__file__ = variables.get("PA_TASK_NAME")
print("BEGIN " + __file__)

# -------------------------------------------------------------
# Import an external python script containing a collection of
# common utility Python functions and classes
PA_CATALOG_REST_URL = variables.get("PA_CATALOG_REST_URL")
PA_PYTHON_UTILS_URL = PA_CATALOG_REST_URL + "/buckets/machine-learning-scripts/resources/Utils/raw"
if PA_PYTHON_UTILS_URL.startswith('https'):
    exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL, context=ssl._create_unverified_context()).read(), globals())
else:
    exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL).read(), globals())
global check_task_is_enabled, preview_dataframe_in_task_result
global is_nvidia_rapids_enabled, is_not_none_not_empty
global raiser, get_input_variables, dict_to_obj
global get_and_decompress_json_dataframe

# -------------------------------------------------------------
# Check if the Python task is enabled or not
check_task_is_enabled()

# -------------------------------------------------------------
# Check if NVIDIA RAPIDS is enabled
# https://rapids.ai/
#
NVIDIA_RAPIDS_ENABLED = is_nvidia_rapids_enabled()
print('NVIDIA_RAPIDS_ENABLED: ', NVIDIA_RAPIDS_ENABLED)

if NVIDIA_RAPIDS_ENABLED:
    import cudf

    def cross_val_score(clf, X, y, cv=3, scoring=None):
        kf = StratifiedKFold(cv)
        acc_scores = []
        i = 0
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            try:
                clf.fit(X_train, y_train)
            except:
                clf.fit(X_train)
            y_pred = clf.predict(X_test)
            acc_score = accuracy_score(y_test, y_pred.to_pandas())
            acc_scores.append(acc_score)
            i += 1
        return acc_scores
else:
    from sklearn.model_selection import cross_val_score

# -------------------------------------------------------------
# Get data from the propagated variables
#
input_variables = {
    'task.dataframe_id': None,
    'task.dataframe_id_train': None,
    'task.algorithm_json': None,
    'task.label_column': None,
    'task.feature_names': None
}
get_input_variables(input_variables)

dataframe_id = None
if input_variables['task.dataframe_id'] is not None:
    dataframe_id = input_variables['task.dataframe_id']
if input_variables['task.dataframe_id_train'] is not None:
    dataframe_id = input_variables['task.dataframe_id_train']
print("dataframe id (in): ", dataframe_id)

dataframe_json = get_and_decompress_json_dataframe(dataframe_id)

if NVIDIA_RAPIDS_ENABLED:
    dataframe = cudf.read_json(dataframe_json, orient='split')
else:
    dataframe = pd.read_json(dataframe_json, orient='split')

is_labeled_data = False
LABEL_COLUMN = variables.get("LABEL_COLUMN")
if is_not_none_not_empty(LABEL_COLUMN):
    is_labeled_data = True
else:
    LABEL_COLUMN = input_variables['task.label_column']
    if is_not_none_not_empty(LABEL_COLUMN):
        is_labeled_data = True

algorithm_json = input_variables['task.algorithm_json']
assert algorithm_json is not None
algorithm = json.loads(algorithm_json)
print("algorithm:\n", algorithm)
alg = dict_to_obj(algorithm)
if not hasattr(alg, 'automl'):
    alg.automl = True
if not hasattr(alg, 'sampling'):
    alg.sampling = False

model = None
if alg.is_supervised:
    # -------------------------------------------------------------
    # Classification algorithms
    #
    if alg.name == 'TPOT_Classifier':
        from tpot import TPOTClassifier
        model = TPOTClassifier(
            generations=alg.generations,
            cv=alg.cv,
            scoring=alg.scoring,
            verbosity=alg.verbosity
        )
    elif alg.name == 'AutoSklearn_Classifier':
        from autosklearn import classification
        if alg.sampling:
            model = classification.AutoSklearnClassifier(
                time_left_for_this_task=alg.task_time,
                per_run_time_limit=alg.run_time,
                resampling_strategy=alg.sampling_strategy,
                resampling_strategy_arguments={'folds': alg.folds}
            )
        else:
            model = classification.AutoSklearnClassifier(
                time_left_for_this_task=alg.task_time,
                per_run_time_limit=alg.run_time
            )
    elif alg.name == 'SupportVectorMachines':
        from sklearn.svm import SVC
        model = SVC(**alg.input_variables.__dict__)
    elif alg.name == 'GaussianNaiveBayes':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB(**alg.input_variables.__dict__)
    elif alg.name == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**alg.input_variables.__dict__)
    elif alg.name == 'AdaBoost' and alg.type == 'classification':
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(**alg.input_variables.__dict__)
    elif alg.name == 'GradientBoosting' and alg.type == 'classification':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(**alg.input_variables.__dict__)
    elif alg.name == 'RandomForest' and alg.type == 'classification':
        if NVIDIA_RAPIDS_ENABLED:
            from cuml.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**alg.input_variables.__dict__)
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**alg.input_variables.__dict__)
    elif alg.name == 'XGBoost' and alg.type == 'classification':
        from xgboost.sklearn import XGBClassifier
        model = XGBClassifier(**alg.input_variables.__dict__)
    elif alg.name == 'CatBoost' and alg.type == 'classification':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**alg.input_variables.__dict__)

    # -------------------------------------------------------------
    # Regression algorithms
    #
    elif alg.name == 'TPOT_Regressor':
        from tpot import TPOTRegressor
        model = TPOTRegressor(
            generations=alg.generations,
            cv=alg.cv,
            scoring=alg.scoring,
            verbosity=alg.verbosity
        )
    elif alg.name == 'AutoSklearn_Regressor':
        from autosklearn import regression
        if alg.sampling:
            model = regression.AutoSklearnRegressor(
                time_left_for_this_task=alg.task_time,
                per_run_time_limit=alg.run_time,
                resampling_strategy=alg.sampling_strategy,
                resampling_strategy_arguments={'folds': alg.folds}
            )
        else:
            model = regression.AutoSklearnRegressor(
                time_left_for_this_task=alg.task_time,
                per_run_time_limit=alg.run_time
            )
    elif alg.name == 'LinearRegression':
        if NVIDIA_RAPIDS_ENABLED:
            from cuml.linear_model import LinearRegression
            model = LinearRegression(**alg.input_variables.__dict__)
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression(**alg.input_variables.__dict__)
    elif alg.name == 'SupportVectorRegression':
        from sklearn.svm import SVR
        model = SVR(**alg.input_variables.__dict__)
    elif alg.name == 'BayesianRidgeRegression':
        from sklearn.linear_model import BayesianRidge
        model = BayesianRidge(**alg.input_variables.__dict__)
    elif alg.name == 'AdaBoost' and alg.type == 'regression':
        from sklearn.ensemble import AdaBoostRegressor
        model = AdaBoostRegressor(**alg.input_variables.__dict__)
    elif alg.name == 'GradientBoosting' and alg.type == 'regression':
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(**alg.input_variables.__dict__)
    elif alg.name == 'RandomForest' and alg.type == 'regression':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**alg.input_variables.__dict__)
    elif alg.name == 'XGBoost' and alg.type == 'regression':
        from xgboost.sklearn import XGBRegressor
        model = XGBRegressor(**alg.input_variables.__dict__)
    elif alg.name == 'CatBoost' and alg.type == 'regression':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(**alg.input_variables.__dict__)
else:
    # -------------------------------------------------------------
    # Anomaly detection algorithms
    #
    if alg.name == 'OneClassSVM':
        from sklearn import svm
        model = svm.OneClassSVM(**alg.input_variables.__dict__)
    elif alg.name == 'IsolationForest':
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(**alg.input_variables.__dict__)
    # -------------------------------------------------------------
    # Clustering algorithms
    #
    elif alg.name == 'MeanShift':
        from sklearn.cluster import MeanShift
        model = MeanShift(**alg.input_variables.__dict__)
    elif alg.name == 'KMeans':
        if NVIDIA_RAPIDS_ENABLED:
            from cuml.cluster import KMeans
            model = KMeans(**alg.input_variables.__dict__)
        else:
            from sklearn.cluster import KMeans
            model = KMeans(**alg.input_variables.__dict__)
# -------------------------------------------------------------
dataframe_label = None
model_explainer = None
loss = 0
if model is not None:
    if is_labeled_data:
        columns = [LABEL_COLUMN]
        dataframe_train = dataframe.drop(columns, axis=1)
        dataframe_label = dataframe[LABEL_COLUMN]
    else:
        dataframe_train = dataframe

    if alg.is_supervised:
        # -------------------------------------------------------------
        # Supervised algorithms
        #
        if NVIDIA_RAPIDS_ENABLED:
            for colname in dataframe_train.columns:
                dataframe_train[colname] = dataframe_train[colname].astype('float32')
            model.fit(dataframe_train, dataframe_label.astype('float32'))
            dataframe_train = dataframe_train.to_pandas()
            dataframe_label = dataframe_label.to_pandas()
        else:
            model.fit(dataframe_train.values, dataframe_label.values.ravel())

        # -------------------------------------------------------------
        # Check if cv score should be calculated for the AutoML workflow
        #
        if alg.automl:
            if alg.type == 'classification':
                scores = cross_val_score(model, dataframe_train.values, dataframe_label.values.ravel(),
                                         cv=int(variables.get("N_SPLITS")), scoring=alg.scoring)
                loss = 1 - np.mean(scores)
                if (not alg.name.startswith("TPOT") and not alg.name.startswith("AutoSklearn")):
                    model_explainer = shap.KernelExplainer(model.predict_proba, dataframe_train)  # feature importance
            if alg.type == 'anomaly':
                scores = cross_val_score(model, dataframe_train.values, dataframe_label.values.ravel(),
                                         cv=int(variables.get("N_SPLITS")), scoring=alg.scoring)
                loss = 1 - np.mean(scores)
                model_explainer = shap.KernelExplainer(model.predict, dataframe_train)  # feature importance
            if alg.type == 'regression':
                scores = cross_val_score(model, dataframe_train.values, dataframe_label.values.ravel(),
                                         cv=int(variables.get("N_SPLITS")), scoring=alg.scoring)
                loss = np.abs(np.mean(scores))
                if alg.name == 'BayesianRidgeRegression' or alg.name == 'LinearRegression':
                    model_explainer = shap.LinearExplainer(model, dataframe_train)
                else:
                    if (not alg.name.startswith("TPOT") and not alg.name.startswith("AutoSklearn")):
                        model_explainer = shap.KernelExplainer(model.predict, dataframe_train)
        # -------------------------------------------------------------
        # Check if sampling is enabled for AutoSklearn
        #
        if alg.sampling:
            model.refit(dataframe_train.values.copy(), dataframe_label.values.ravel().copy())
        # -------------------------------------------------------------
        # Get the fitted model from TPOT
        #
        if alg.name == 'TPOT_Regressor' or alg.name == 'TPOT_Classifier':
            model = model.fitted_pipeline_
    else:
        # -------------------------------------------------------------
        # Non-supervised algorithms
        #
        if NVIDIA_RAPIDS_ENABLED:
            for colname in dataframe_train.columns:
                dataframe_train[colname] = dataframe_train[colname].astype('float32')
            model.fit(dataframe_train)
            dataframe_train = dataframe_train.to_pandas()
            dataframe_label = dataframe_label.to_pandas() if dataframe_label is not None else None
        else:
            model.fit(dataframe_train.values)
            model_explainer = shap.KernelExplainer(model.predict, dataframe_train)
        if is_labeled_data and alg.automl:
            scores = cross_val_score(model, dataframe_train.values, dataframe_label.values.ravel(),
                                     cv=int(variables.get("N_SPLITS")), scoring=alg.scoring)
            loss = 1 - np.mean(scores)
    # -------------------------------------------------------------
    # Dumps and compress the model
    #
    model_bin = pickle.dumps(model)
    model_compressed = bz2.compress(model_bin)
    model_id = str(uuid.uuid4())
    variables.put(model_id, model_compressed)
    print("model id: ", model_id)
    print('model size (original):   ', sys.getsizeof(model_bin), " bytes")
    print('model size (compressed): ', sys.getsizeof(model_compressed), " bytes")
    resultMetadata.put("task.model_id", model_id)
else:
    raiser("Algorithm not found!")

# -----------------------------------------------------------------
# Data drift measures
#
mean_df_train = dataframe_train.mean(axis=0)  # mean
std_df_train = dataframe_train.std(axis=0)  # standard deviation
dataframe_model_metadata = pd.DataFrame({0: mean_df_train, 1: std_df_train}).T

dataframe = dataframe.to_pandas() if NVIDIA_RAPIDS_ENABLED else dataframe
dataframe_model_metadata = dataframe_model_metadata.to_pandas() if NVIDIA_RAPIDS_ENABLED else dataframe_model_metadata

dataframe_json = dataframe.to_json(orient='split').encode()
# dataframe_model_meta_json = dataframe_model_metadata.to_json(orient='split').encode()
dataframe_model_meta_json = dataframe_model_metadata.to_json(orient='values').encode()

compressed_data = bz2.compress(dataframe_json)
compressed_model_metadata = bz2.compress(dataframe_model_meta_json)

dataframe_id = str(uuid.uuid4())
variables.put(dataframe_id, compressed_data)

model_metadata_id = str(uuid.uuid4())
variables.put(model_metadata_id, compressed_model_metadata)

print("dataframe id (out): ", dataframe_id)
print("model metadata id (out): ", model_metadata_id)

resultMetadata.put("task.name", __file__)
resultMetadata.put("task.algorithm_json", algorithm_json)
resultMetadata.put("task.label_column", LABEL_COLUMN)
resultMetadata.put("task.model_metadata_id", model_metadata_id)
resultMetadata.put("task.dataframe_id", dataframe_id)
resultMetadata.put("task.feature_names", input_variables['task.feature_names'])
# -----------------------------------------------------------------
# Explainer shap
#
if model_explainer:
    model_explainer_bin = pickle.dumps(model_explainer)
    model_explainer_compressed = bz2.compress(model_explainer_bin)  # explainer object
    model_explainer_id = str(uuid.uuid4())
    variables.put(model_explainer_id, model_explainer_compressed)
    resultMetadata.put("task.model_explainer_id", model_explainer_id)
# -----------------------------------------------------------------

token = variables.get("TOKEN")
# Convert from JSON to dict
token = json.loads(token)

# return the loss value
result_map = {
    'token': token,
    'loss': loss
}

result_map = json.dumps(result_map)
resultMap.put("RESULT_JSON", result_map)
print('result_map: ', result_map)

# -------------------------------------------------------------
# Preview results
#
preview_dataframe_in_task_result(dataframe)

# -------------------------------------------------------------
print("END " + __file__)
