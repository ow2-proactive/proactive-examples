"""Proactive Train Model for Machine Learning

This module contains the Python script for the Train Model task.
"""
import ssl
import urllib.request
import json
import shap

import numpy as np
import pandas as pd

# from packaging import version

global variables, resultMetadata, resultMap

__file__ = variables.get("PA_TASK_NAME")
print("BEGIN " + __file__)

# -------------------------------------------------------------
# Get schedulerapi access and acquire session id
schedulerapi.connect()
sessionid = schedulerapi.getSession()

# -------------------------------------------------------------
# Import an external python script containing a collection of
# common utility Python functions and classes
PA_CATALOG_REST_URL = variables.get("PA_CATALOG_REST_URL")
PA_PYTHON_UTILS_URL = PA_CATALOG_REST_URL + "/buckets/ai-machine-learning/resources/Utils_Script/raw"
req = urllib.request.Request(PA_PYTHON_UTILS_URL)
req.add_header('sessionid', sessionid)
if PA_PYTHON_UTILS_URL.startswith('https'):
    content = urllib.request.urlopen(req, context=ssl._create_unverified_context()).read()
else:
    content = urllib.request.urlopen(req).read()
exec(content, globals())
global check_task_is_enabled, is_nvidia_rapids_enabled
global raiser, get_input_variables, dict_to_obj, is_not_none_not_empty
global get_and_decompress_json_dataframe, preview_dataframe_in_task_result
global compress_and_transfer_dataframe, compress_and_transfer_data, apply_encoder

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
    # import cuml
    # import cupy as cp

    def cross_val_score(clf, X, y, cv=3, scoring=None):
        from sklearn.model_selection import StratifiedKFold
        # from sklearn.model_selection import KFold
        from cupy import asnumpy
        from cuml.metrics import accuracy_score
        kf = StratifiedKFold(cv)
        # kf = KFold(cv)
        acc_scores = []
        i = 0
        # for train_index, test_index in kf.split(X):
        for train_index, test_index in kf.split(X, asnumpy(y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # X_train, X_test = X[train_index], X[test_index]
            # y_train, y_test = y[train_index], y[test_index]
            try:
                clf.fit(X_train, y_train, convert_dtype=True)
            except:
                clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc_score = accuracy_score(y_test, y_pred)
            acc_scores.append(acc_score)
            i += 1
        return acc_scores
else:
    from sklearn.model_selection import cross_val_score


def warn_not_gpu_support(alg):
    if NVIDIA_RAPIDS_ENABLED:
        print(alg.name + " not supported on GPU!")


# -------------------------------------------------------------
# Get data from the propagated variables
#
input_variables = {
    'task.dataframe_id': None,
    'task.dataframe_id_train': None,
    'task.algorithm_json': None,
    'task.label_column': None,
    'task.feature_names': None,
    'task.encode_map_json': None
}
get_input_variables(input_variables)

dataframe_id = None
if input_variables['task.dataframe_id'] is not None:
    dataframe_id = input_variables['task.dataframe_id']
if input_variables['task.dataframe_id_train'] is not None:
    dataframe_id = input_variables['task.dataframe_id_train']
print("dataframe id (in): ", dataframe_id)

encode_map_json = input_variables['task.encode_map_json']
encode_map = None
if encode_map_json is not None:
    encode_map = json.loads(encode_map_json)

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
        warn_not_gpu_support(alg)
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
        warn_not_gpu_support(alg)
    elif alg.name == 'SupportVectorMachines':
        if NVIDIA_RAPIDS_ENABLED:
            from cuml.svm import SVC
        else:
            from sklearn.svm import SVC
        model = SVC(**alg.input_variables.__dict__)
    elif alg.name == 'GaussianNaiveBayes':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB(**alg.input_variables.__dict__)
        warn_not_gpu_support(alg)
    elif alg.name == 'LogisticRegression':
        if NVIDIA_RAPIDS_ENABLED:
            # nvidia rapids version should be higher than v0.13
            # if version.parse(cuml.__version__) > version.parse("0.13"):
            from cuml.linear_model import LogisticRegression
        else:
            from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**alg.input_variables.__dict__)
    elif alg.name == 'AdaBoost' and alg.type == 'classification':
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(**alg.input_variables.__dict__)
        warn_not_gpu_support(alg)
    elif alg.name == 'GradientBoosting' and alg.type == 'classification':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(**alg.input_variables.__dict__)
        warn_not_gpu_support(alg)
    elif alg.name == 'RandomForest' and alg.type == 'classification':
        if NVIDIA_RAPIDS_ENABLED:
            from cuml.ensemble import RandomForestClassifier
        else:
            from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**alg.input_variables.__dict__)
    elif alg.name == 'XGBoost' and alg.type == 'classification':
        from xgboost.sklearn import XGBClassifier
        """
        Note from NVIDIA RAPIDS >= 0.17 (no error for == 0.13)
        ValueError: The option use_label_encoder=True is incompatible with inputs of type cuDF or cuPy. 
        Please set use_label_encoder=False when constructing XGBClassifier object. 
        NOTE: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. 
        To remove this warning, do the following: 
        1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 
        2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1]
        """
        if NVIDIA_RAPIDS_ENABLED:
            model = XGBClassifier(**alg.input_variables.__dict__, use_label_encoder=False, tree_method="gpu_hist")
        else:
            model = XGBClassifier(**alg.input_variables.__dict__)
    elif alg.name == 'CatBoost' and alg.type == 'classification':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**alg.input_variables.__dict__)
        warn_not_gpu_support(alg)

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
        warn_not_gpu_support(alg)
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
        warn_not_gpu_support(alg)
    elif alg.name == 'LinearRegression':
        if NVIDIA_RAPIDS_ENABLED:
            from cuml.linear_model import LinearRegression
            model = LinearRegression(**alg.input_variables.__dict__)
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression(**alg.input_variables.__dict__)
    elif alg.name == 'SupportVectorRegression':
        if NVIDIA_RAPIDS_ENABLED:
            from cuml.svm import SVR
        else:
            from sklearn.svm import SVR
        model = SVR(**alg.input_variables.__dict__)
    elif alg.name == 'BayesianRidgeRegression':
        from sklearn.linear_model import BayesianRidge
        model = BayesianRidge(**alg.input_variables.__dict__)
        warn_not_gpu_support(alg)
    elif alg.name == 'AdaBoost' and alg.type == 'regression':
        from sklearn.ensemble import AdaBoostRegressor
        model = AdaBoostRegressor(**alg.input_variables.__dict__)
        warn_not_gpu_support(alg)
    elif alg.name == 'GradientBoosting' and alg.type == 'regression':
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(**alg.input_variables.__dict__)
        warn_not_gpu_support(alg)
    elif alg.name == 'RandomForest' and alg.type == 'regression':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**alg.input_variables.__dict__)
        warn_not_gpu_support(alg)
    elif alg.name == 'XGBoost' and alg.type == 'regression':
        from xgboost.sklearn import XGBRegressor
        model = XGBRegressor(**alg.input_variables.__dict__)
    elif alg.name == 'CatBoost' and alg.type == 'regression':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(**alg.input_variables.__dict__)
        warn_not_gpu_support(alg)
else:
    # -------------------------------------------------------------
    # Anomaly detection algorithms
    #
    if alg.name == 'OneClassSVM':
        from sklearn import svm
        model = svm.OneClassSVM(**alg.input_variables.__dict__)
        warn_not_gpu_support(alg)
    elif alg.name == 'IsolationForest':
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(**alg.input_variables.__dict__)
        warn_not_gpu_support(alg)
    # -------------------------------------------------------------
    # Clustering algorithms
    #
    elif alg.name == 'MeanShift':
        from sklearn.cluster import MeanShift
        model = MeanShift(**alg.input_variables.__dict__)
        warn_not_gpu_support(alg)
    elif alg.name == 'KMeans':
        if NVIDIA_RAPIDS_ENABLED:
            from cuml.cluster import KMeans
            model = KMeans(**alg.input_variables.__dict__)
        else:
            from sklearn.cluster import KMeans
            model = KMeans(**alg.input_variables.__dict__)
    # -------------------------------------------------------------
    # Drift algorithms
    #
    if alg.name == 'KolmogorovSmirnov':
        from alibi_detect.cd import KSDrift
        x_ref = dataframe.drop([LABEL_COLUMN], axis=1).values
        data_ref = x_ref.astype('float32')
        model = KSDrift(x_ref=data_ref, **alg.input_variables.__dict__)
    elif alg.name == 'SpotDiff':
        from alibi_detect.cd import SpotTheDiffDrift
        x_ref = dataframe.drop([LABEL_COLUMN], axis=1).values
        data_ref = x_ref.astype('float32')
        model = SpotTheDiffDrift(x_ref=data_ref, **alg.input_variables.__dict__)
# -------------------------------------------------------------
dataframe_train = None
dataframe_label = None
model_explainer = None
loss = 0
if model is not None:
    print('-' * 30)
    print(model)
    print('-' * 30)

    if is_labeled_data:
        dataframe_train = dataframe.drop([LABEL_COLUMN], axis=1)
        dataframe_label = dataframe[LABEL_COLUMN]
    else:
        dataframe_train = dataframe

    if NVIDIA_RAPIDS_ENABLED:
        for colname in dataframe_train.columns:
            dataframe_train[colname] = dataframe_train[colname].astype('float32')
        dataframe_label = dataframe_label.astype('float32') if dataframe_label is not None else None
    #     X_train = dataframe_train.as_gpu_matrix()
    #     X_label = cp.asarray(dataframe_label) if dataframe_label is not None else None
    # else:
    #     X_train = dataframe_train.values
    #     X_label = dataframe_label.values.ravel() if dataframe_label is not None else None

    if alg.is_supervised:
        # -------------------------------------------------------------
        # Supervised algorithms
        #
        try:
            model.fit(dataframe_train, dataframe_label, convert_dtype=True)
        except:
            model.fit(dataframe_train, dataframe_label)
        # -------------------------------------------------------------
        # Check if cv score should be calculated for the AutoML workflow
        #
        if alg.automl:
            if NVIDIA_RAPIDS_ENABLED and alg.type == 'regression':
                print("[Warning] The loss could not be calculated using cross_val_score on GPU for regression "
                      "algorithms!")
            else:
                CV = int(variables.get("N_SPLITS")) if variables.get("N_SPLITS") is not None else 5
                scores = cross_val_score(model, dataframe_train, dataframe_label, cv=CV, scoring=alg.scoring)
                if alg.type == 'classification' or alg.type == 'anomaly':
                    loss = 1 - np.mean(scores)
                if alg.type == 'regression':
                    loss = np.abs(np.mean(scores))
        # -------------------------------------------------------------
        # Check if model explainer can be used
        #
        if not NVIDIA_RAPIDS_ENABLED:
            if alg.name == 'BayesianRidgeRegression' or alg.name == 'LinearRegression':
                model_explainer = shap.LinearExplainer(model, dataframe_train)
            else:
                if (not alg.name.startswith("TPOT") and
                    not alg.name.startswith("AutoSklearn") and
                    not alg.name.startswith("XGBoost")):
                    if alg.type == 'classification':
                        model_explainer = shap.KernelExplainer(model.predict_proba, dataframe_train)
                    if alg.type == 'regression' or alg.type == 'anomaly':
                        model_explainer = shap.KernelExplainer(model.predict, dataframe_train)
                else:
                    print("Model explainer not supported for the selected algorithm!")
        else:
            print("Model explainer not supported on GPU!")
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
    elif alg.type == 'drift_detection':
        from alibi_detect.saving import save_detector
        save_detector(model, '.')
    else:
        # -------------------------------------------------------------
        # Non-supervised algorithms
        #
        if NVIDIA_RAPIDS_ENABLED:
            model.fit(dataframe_train)
        else:
            model.fit(dataframe_train.values)
            model_explainer = shap.KernelExplainer(model.predict, dataframe_train)
        # if is_labeled_data and alg.automl:
        #    scores = cross_val_score(model, dataframe_train.values, dataframe_label.values.ravel(),
        #                             cv=int(variables.get("N_SPLITS")), scoring=alg.scoring)
        #    loss = 1 - np.mean(scores)
    # -------------------------------------------------------------
    # Dumps and compress the model
    #
    if alg.type != 'drift_detection':
        model_id = compress_and_transfer_data(model, "model")
        resultMetadata.put("task.model_id", model_id)
else:
    raiser("Algorithm not found!")
    # Example usage
print("Current directory contents:", os.listdir('.'))
# -------------------------------------------------------------
# Transfer data to the next tasks
#
if NVIDIA_RAPIDS_ENABLED:
    dataframe = dataframe.to_pandas()
    dataframe_train = dataframe_train.to_pandas() if dataframe_train is not None else None
    dataframe_label = dataframe_label.to_pandas() if dataframe_label is not None else None

dataframe_id = compress_and_transfer_dataframe(dataframe)
print("dataframe id (out): ", dataframe_id)

# -----------------------------------------------------------------
# Data drift measures
#
# [deprecated]
# import warnings
# warnings.warn("model_metadata is deprecated", DeprecationWarning)
# mean_df_train = dataframe_train.mean(axis=0)  # mean
# std_df_train = dataframe_train.std(axis=0)  # standard deviation
# dataframe_model_metadata = pd.DataFrame({0: mean_df_train, 1: std_df_train}).T
# model_metadata_id = compress_and_transfer_dataframe(dataframe_model_metadata, orient='values')
# print("model metadata id (out): ", model_metadata_id)
# resultMetadata.put("task.model_metadata_id", model_metadata_id)

resultMetadata.put("task.name", __file__)
resultMetadata.put("task.algorithm_json", algorithm_json)
resultMetadata.put("task.dataframe_id", dataframe_id)
resultMetadata.put("task.label_column", LABEL_COLUMN)
resultMetadata.put("task.feature_names", input_variables['task.feature_names'])
resultMetadata.put("task.encode_map_json", input_variables['task.encode_map_json'])

# -----------------------------------------------------------------
# Model Explainer
#
if model_explainer:
    model_explainer_id = compress_and_transfer_data(model, "model_explainer")
    resultMetadata.put("task.model_explainer_id", model_explainer_id)

# -----------------------------------------------------------------
# For AutoML
#
token = variables.get("TOKEN")
token = json.loads(token)
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
if encode_map is not None and is_labeled_data:
    # apply_encoder(dataframe, columns, encode_map, sep=",")
    dataframe_aux = apply_encoder(dataframe, LABEL_COLUMN, encode_map)
    preview_dataframe_in_task_result(dataframe_aux)
else:
    preview_dataframe_in_task_result(dataframe)

# -------------------------------------------------------------
print("END " + __file__)