__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() == 'false':
    print("Task " + __file__ + " disabled")
    quit()

print("BEGIN " + __file__)

import bz2
import sys
import uuid

import pandas as pd


def scale_columns(df, columns, scaler_name="RobustScaler"):
  from sklearn import preprocessing

  scaler = None
  if scaler_name == "StandardScaler":
    scaler = preprocessing.StandardScaler()
  if scaler_name == "RobustScaler":
    scaler = preprocessing.RobustScaler()
  if scaler_name == "MinMaxScaler":
    scaler = preprocessing.MinMaxScaler()
  if scaler_name == "Normalizer":
    scaler = preprocessing.Normalizer()
  assert scaler is not None

  data = df.filter(columns, axis=1)
  print(scaler.fit(data))

  scaled_data = scaler.transform(data)
  scaled_df = pd.DataFrame(scaled_data, columns=columns)

  dataframe_scaled = df.copy()
  dataframe_scaled = dataframe_scaled.reset_index(drop=True)
  for column in columns:
    dataframe_scaled[column] = scaled_df[column]

  return dataframe_scaled, scaler


def apply_scaler(df, columns, scaler):
  data = df.filter(columns, axis=1)
  scaled_data = scaler.transform(data)
  scaled_df = pd.DataFrame(scaled_data, columns=columns)

  dataframe_scaled = df.copy()
  dataframe_scaled = dataframe_scaled.reset_index(drop=True)
  for column in columns:
    dataframe_scaled[column] = scaled_df[column]

  return dataframe_scaled


# -------------------------------------------------------------
# Get data from the propagated variables
#
COLUMNS_NAME = variables.get("COLUMNS_NAME")
SCALER_NAME = variables.get("SCALER_NAME")

assert COLUMNS_NAME is not None and COLUMNS_NAME is not ""
assert SCALER_NAME is not None and SCALER_NAME is not ""

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

dataframe = pd.read_json(dataframe_json, orient='split')

columns = [x.strip() for x in COLUMNS_NAME.split(',')]
dataframe, scaler = scale_columns(dataframe, columns, SCALER_NAME)
print(dataframe.head())

dataframe_json = dataframe.to_json(orient='split').encode()
compressed_data = bz2.compress(dataframe_json)

dataframe_id = str(uuid.uuid4())
variables.put(dataframe_id, compressed_data)

print("dataframe id (out): ", dataframe_id)
print('dataframe size (original):   ', sys.getsizeof(dataframe_json), " bytes")
print('dataframe size (compressed): ', sys.getsizeof(compressed_data), " bytes")
print(dataframe.head())

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

print("END " + __file__)
