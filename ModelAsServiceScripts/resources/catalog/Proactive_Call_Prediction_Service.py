import requests, json, bz2
import pandas as pd

def raiser(msg): raise Exception(msg)

# Get variables
API_PREDICT = variables.get("PREDICT_EXTENSION") if variables.get("PREDICT_EXTENSION") else raiser("PREDICT_EXTENSION is None")
LABEL_COLUMN = variables.get("LABEL_COLUMN") if variables.get("LABEL_COLUMN") else raiser("LABEL_COLUMN is None")

SERVICE_TOKEN = variables.get("SERVICE_TOKEN") if variables.get("SERVICE_TOKEN") else variables.get("SERVICE_TOKEN_PROPAGATED")
assert SERVICE_TOKEN is not None

API_ENDPOINT = variables.get("PREDICT_MODEL_ENDPOINT") if variables.get("PREDICT_MODEL_ENDPOINT") else variables.get("ENDPOINT_MODEL")
assert API_ENDPOINT is not None

API_PREDICT_ENDPOINT = API_ENDPOINT + API_PREDICT
print("API_PREDICT_ENDPOINT: ", API_PREDICT_ENDPOINT)

# Get data from previous tasks
input_variables = {
  'task.dataframe_id_test': None,
  'task.dataframe_id': None
}
for key in input_variables.keys():
    for res in results:
        value = res.getMetadata().get(key)
        if value is not None:
            input_variables[key] = value
            break

if input_variables['task.dataframe_id_test'] is not None:
    dataframe_id = input_variables['task.dataframe_id_test']
    dataframe_json = variables.get(dataframe_id)
    dataframe_json = bz2.decompress(dataframe_json).decode()
    assert dataframe_json is not None
    dataframe = pd.read_json(dataframe_json, orient='split')
    dataframe_test = dataframe.drop([LABEL_COLUMN], axis=1, inplace=False)
    dataframe_json = dataframe_test.to_json(orient='values')
elif input_variables['task.dataframe_id'] is not None:
    dataframe_id = input_variables['task.dataframe_id']
    dataframe_json = variables.get(dataframe_id)
    dataframe_json = bz2.decompress(dataframe_json).decode()
    assert dataframe_json is not None
    dataframe = pd.read_json(dataframe_json, orient='split')
    dataframe_test = dataframe.drop([LABEL_COLUMN], axis=1, inplace=False)
    dataframe_json = dataframe_test.to_json(orient='values')
elif variables.get("INPUT_DATA") is not None:
    dataframe_json = variables.get("INPUT_DATA")
else:
    raiser("There is no input data")

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
data = {'dataframe_json': dataframe_json, 'api_token': SERVICE_TOKEN}
data_json = json.dumps(data)
req = requests.post(API_PREDICT_ENDPOINT, data=data_json, headers=headers)

predictions = json.loads(req.text)
print("predictions:\n", predictions)

predictions = pd.read_json(predictions, orient='records')
df_dataframe = pd.read_json(dataframe_json, orient='records')
dataframe = df_dataframe.assign(predictions=predictions)
    
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
