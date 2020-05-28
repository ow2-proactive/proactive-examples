__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() == 'false':
    print("Task " + __file__ + " disabled")
    quit()

print("BEGIN " + __file__)

import bz2
import sys
import uuid
import pandas as pd

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

# -------------------------------------------------------------
# Get data from the propagated variables
#
time_column = variables.get("TIME_COLUMN")
ref_column = variables.get("REF_COLUMN")
all_features = variables.get("ALL_FEATURES")

input_variables = {
    'task.dataframe_id': None,
    'task.label_column': None
}
for key in input_variables.keys():
    for res in results:
        value = res.getMetadata().get(key)
        if value is not None:
            input_variables[key] = value
            break

dataframe_id = input_variables['task.dataframe_id']
print("dataframe id (in): ", dataframe_id)

dataframe_json = variables.get(dataframe_id)
assert dataframe_json is not None
dataframe_json = bz2.decompress(dataframe_json).decode()

dataframe_df = pd.read_json(dataframe_json, orient='split')

# Add the list of features that need to be extracted 
# Check the full list of features on this link:
# http://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
if all_features == "True":
    extraction_settings = {
        "length": None,
        "absolute_sum_of_changes": None,
        "abs_energy": None,
        # "sample_entropy": None,
        "number_peaks": [{"n": 2}],
        "number_cwt_peaks": [{"n": 2}, {"n": 3}],
        "autocorrelation": [{"lag": 2}, {"lag": 3}]
        # "value_count": #"large_standard_deviation": [{"r": 0.05}, {"r": 0.1}]
    }
# For convenience, three dictionaries are predefined and can be used right away
# ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
# MinimalFCParameters is set by default
else:
    extraction_settings = MinimalFCParameters()

extracted_features = extract_features(dataframe_df,
                                      column_id=ref_column,
                                      column_sort=time_column,
                                      default_fc_parameters=extraction_settings)
extracted_features[ref_column] = extracted_features.index

dataframe_json = extracted_features.to_json(orient='split').encode()
compressed_data = bz2.compress(dataframe_json)

dataframe_id = str(uuid.uuid4())
variables.put(dataframe_id, compressed_data)

print("dataframe id: ", dataframe_id)
print('dataframe size (original):   ', sys.getsizeof(dataframe_json), " bytes")
print('dataframe size (compressed): ', sys.getsizeof(compressed_data), " bytes")

resultMetadata.put("task.name", __file__)
resultMetadata.put("task.dataframe_id", dataframe_id)
resultMetadata.put("task.label_column", input_variables['task.label_column'])

# -------------------------------------------------------------
# Preview results
#
LIMIT_OUTPUT_VIEW = variables.get("LIMIT_OUTPUT_VIEW")
LIMIT_OUTPUT_VIEW = 5 if LIMIT_OUTPUT_VIEW is None else int(LIMIT_OUTPUT_VIEW)
if LIMIT_OUTPUT_VIEW > 0:
    print("task result limited to: ", LIMIT_OUTPUT_VIEW, " rows")
    dataframe = dataframe.head(LIMIT_OUTPUT_VIEW).copy()
result = ''
with pd.option_context('display.max_colwidth', -1):
    result = dataframe.to_html(escape=False, classes='table table-bordered table-striped', justify='center')
result = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Machine Learning Preview</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" 
integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
</head>
<body class="container">
<h1 class="text-center my-4" style="color:#003050;">Data Preview</h1>
<div style="text-align:center">{0}</div>
</body></html>""".format(result)
result = result.encode('utf-8')
resultMetadata.put("file.extension", ".html")
resultMetadata.put("file.name", "output.html")
resultMetadata.put("content.type", "text/html")
# -------------------------------------------------------------

print("END " + __file__)
