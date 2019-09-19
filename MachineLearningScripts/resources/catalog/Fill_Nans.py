__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import sys, bz2, uuid, json
import pandas as pd
import numpy as np

input_variables = {'task.dataframe_id': None}
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
dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

FILL_MAP = variables.get("FILL_MAP")
if FILL_MAP is not None and FILL_MAP is not "":
  fill_map = json.loads(FILL_MAP)
  dataframe.fillna(value=fill_map, inplace=True)
else:
  dataframe.fillna(0, inplace=True)

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

#============================== Preview results ===============================

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
                  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
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