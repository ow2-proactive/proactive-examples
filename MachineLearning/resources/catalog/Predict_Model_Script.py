# Copyright Activeeon 2007-2023. All rights reserved.
"""Proactive Predict Model for Machine Learning

This module contains the Python script for the Predict Model task.
"""
import ssl
import urllib.request
import bz2
import json
import pickle
import sys
import uuid
import numpy as np
import pandas as pd
import xml.sax.saxutils as saxutils

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mutual_info_score
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score

from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score

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
global check_task_is_enabled, preview_dataframe_in_task_result
global is_nvidia_rapids_enabled, is_not_none_not_empty
global raiser, get_input_variables, dict_to_obj, apply_encoder
global get_and_decompress_json_dataframe, compress_and_transfer_dataframe

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
    from cudf import read_json
else:
    from pandas import read_json

# -------------------------------------------------------------
# Get data from the propagated variables
#
input_variables = {
    'task.dataframe_id': None,
    'task.dataframe_id_test': None,
    'task.algorithm_json': None,
    'task.label_column': None,
    'task.model_id': None,
    'task.encode_map_json': None
}
get_input_variables(input_variables)

dataframe_id = None
if input_variables['task.dataframe_id'] is not None:
    dataframe_id = input_variables['task.dataframe_id']
if input_variables['task.dataframe_id_test'] is not None:
    dataframe_id = input_variables['task.dataframe_id_test']
print("dataframe id (in): ", dataframe_id)

encode_map_json = input_variables['task.encode_map_json']
encode_map = None
if encode_map_json is not None:
    encode_map = json.loads(encode_map_json)

dataframe_json = get_and_decompress_json_dataframe(dataframe_id)
dataframe = read_json(dataframe_json, orient='split')

is_labeled_data = False
LABEL_COLUMN = variables.get("LABEL_COLUMN")
if is_not_none_not_empty(LABEL_COLUMN):
    is_labeled_data = True
else:
    LABEL_COLUMN = input_variables['task.label_column']
    if is_not_none_not_empty(LABEL_COLUMN):
        is_labeled_data = True

model_id = input_variables['task.model_id']
model_compressed = variables.get(model_id)
model_bin = bz2.decompress(model_compressed)
assert model_bin is not None
print("model id (in): ", model_id)
print("model size: ", sys.getsizeof(model_compressed), " bytes")
print("model size (decompressed): ", sys.getsizeof(model_bin), " bytes")

algorithm_json = input_variables['task.algorithm_json']
assert algorithm_json is not None
algorithm = json.loads(algorithm_json)
print("algorithm:\n", algorithm)
alg = dict_to_obj(algorithm)

dataframe_predictions = None
loaded_model = pickle.loads(model_bin)
if loaded_model is not None:
    print('-' * 30)
    print(loaded_model)
    print('-' * 30)

if is_labeled_data:
    dataframe_test = dataframe.drop([LABEL_COLUMN], axis=1)
    dataframe_label = dataframe[LABEL_COLUMN]
    # -------------------------------------------------------------
    # Perform predictions
    #
    if NVIDIA_RAPIDS_ENABLED:
        for colname in dataframe_test.columns:
            dataframe_test[colname] = dataframe_test[colname].astype('float32')
        dataframe_label = dataframe_label.astype('float32')
    try:
        predictions = list(loaded_model.predict(dataframe_test))
    except:
        predictions = list(loaded_model.predict(dataframe_test).to_pandas())
    # -------------------------------------------------------------
    # Convert to Pandas if NVIDIA_RAPIDS_ENABLED = True
    #
    dataframe = dataframe.to_pandas() if NVIDIA_RAPIDS_ENABLED else dataframe
    dataframe_test = dataframe_test.to_pandas() if NVIDIA_RAPIDS_ENABLED else dataframe_test
    dataframe_label = dataframe_label.to_pandas() if NVIDIA_RAPIDS_ENABLED else dataframe_label
    # -------------------------------------------------------------
    # Store predictions on pandas dataframe
    #
    dataframe_predictions = pd.DataFrame(predictions)
    dataframe = dataframe.assign(predictions=dataframe_predictions)
    # -------------------------------------------------------------
    # Convert anomaly outputs to '1's and '0's
    #
    if alg.type == 'anomaly':
        pred_map = {-1: 1, 1: 0}
        dataframe["predictions"].replace(pred_map, inplace=True)
        predictions = dataframe["predictions"].tolist()
    # -------------------------------------------------------------
    # Score model if not clustering and not anomaly
    #
    # if alg.type != 'clustering' and alg.type != 'anomaly':
    #     score = loaded_model.score(dataframe_test, dataframe_label)
    #     print("MODEL SCORE: %.2f" % score)
    # -------------------------------------------------------------
    # CLASSIFICATION AND ANOMALY DETECTION SCORE
    #
    if alg.type == 'classification' or alg.type == 'anomaly':
        reponse_good = '&#9989;'
        reponse_bad = '&#10060;'
        dataframe['results'] = np.where((dataframe[LABEL_COLUMN] == dataframe['predictions']),
                                        saxutils.unescape(reponse_good), saxutils.unescape(reponse_bad))
        accuracy_score_result = accuracy_score(dataframe_label.values.ravel(), predictions)
        precision_score_result = precision_score(dataframe_label.values.ravel(), predictions, average='micro')
        confusion_matrix_result = confusion_matrix(dataframe_label.values.ravel(), predictions)
        print("********************** CLASSIFICATION SCORE **********************")
        print("ACCURACY SCORE:  %.2f" % accuracy_score_result)
        print("CONFUSION MATRIX:\n%s" % confusion_matrix_result)
        print("*******************************************************************")
        resultMap.put("ACCURACY_SCORE", "%.2f"%accuracy_score_result)
        resultMap.put("CONFUSION_MATRIX", confusion_matrix_result.tolist())
    # -------------------------------------------------------------
    # REGRESSION SCORE
    #
    if alg.type == 'regression':
        dataframe['absolute_error'] = dataframe[LABEL_COLUMN] - dataframe['predictions']
        mean_squared_error_result = mean_squared_error(dataframe_label.values.ravel(), predictions)
        mean_absolute_error_result = mean_absolute_error(dataframe_label.values.ravel(), predictions)
        r2_score_result = r2_score(dataframe_label.values.ravel(), predictions)
        print("********************** REGRESSION SCORES **********************")
        print("MEAN SQUARED ERROR: %.2f" % mean_squared_error_result)
        print("MEAN ABSOLUTE ERROR: %.2f" % mean_absolute_error_result)
        print("R2 SCORE: %.2f" % r2_score_result)
        print("***************************************************************")
        resultMap.put("MEAN_SQUARED_ERROR", "%.2f"%mean_squared_error_result)
        resultMap.put("MEAN_ABSOLUTE_ERROR", "%.2f"%mean_absolute_error_result)
        resultMap.put("R2_SCORE", "%.2f"%r2_score_result)
    # -------------------------------------------------------------
    # CLUSTERING SCORE
    #
    if alg.type == 'clustering':
        adjusted_mutual_info_score_result = adjusted_mutual_info_score(dataframe_label.values.ravel(), predictions)
        completeness_score_result = completeness_score(dataframe_label.values.ravel(), predictions)
        homogeneity_score_result = homogeneity_score(dataframe_label.values.ravel(), predictions)
        mutual_info_score_result = mutual_info_score(dataframe_label.values.ravel(), predictions)
        v_measure_score_result = v_measure_score(dataframe_label.values.ravel(), predictions)
        print("********************** CLUSTERING SCORES **********************")
        print("ADJUSTED MUTUAL INFORMATION: %.2f" % adjusted_mutual_info_score_result)
        print("COMPLETENESS SCORE: %.2f" % completeness_score_result)
        print("HOMOGENEITY METRIC: %.2f" % homogeneity_score_result)
        print("MUTUAL INFORMATION: %.2f" % mutual_info_score_result)
        print("V-MEASURE CLUSTER MEASURE: %.2f" % v_measure_score_result)
        print("***************************************************************")
        resultMap.put("ADJUSTED_MUTUAL_INFORMATION", "%.2f"%adjusted_mutual_info_score_result)
        resultMap.put("COMPLETENESS_SCORE", "%.2f"%completeness_score_result)
        resultMap.put("HOMOGENEITY_METRIC", "%.2f"%homogeneity_score_result)
        resultMap.put("MUTUAL_INFORMATION", "%.2f"%mutual_info_score_result)
        resultMap.put("V-MEASURE_CLUSTER_MEASURE", "%.2f"%v_measure_score_result)
else:
    if NVIDIA_RAPIDS_ENABLED:
        for colname in dataframe:
            dataframe[colname] = dataframe[colname].astype('float32')
    predictions = list(loaded_model.predict(dataframe))
    dataframe_predictions = pd.DataFrame(predictions)
    dataframe = dataframe.to_pandas() if NVIDIA_RAPIDS_ENABLED else dataframe
    dataframe = dataframe.assign(predictions=dataframe_predictions)


# -------------------------------------------------------------
# Transfer data to the next tasks
#
dataframe_id = compress_and_transfer_dataframe(dataframe)
print("dataframe id (out): ", dataframe_id)

resultMetadata.put("task.name", __file__)
resultMetadata.put("task.dataframe_id", dataframe_id)
resultMetadata.put("task.algorithm_json", algorithm_json)
resultMetadata.put("task.label_column", LABEL_COLUMN)
resultMetadata.put("task.encode_map_json", input_variables['task.encode_map_json'])

# -------------------------------------------------------------
# Preview results
#
if encode_map is not None and is_labeled_data:
    # apply_encoder(dataframe, columns, encode_map, sep=",")
    encode_map['predictions'] = encode_map[LABEL_COLUMN]
    dataframe_aux = apply_encoder(dataframe, [LABEL_COLUMN, 'predictions'], encode_map)
    preview_dataframe_in_task_result(dataframe_aux)
else:
    preview_dataframe_in_task_result(dataframe)

# -------------------------------------------------------------
print("END " + __file__)