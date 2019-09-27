import requests
import json, bz2
import numpy as np
import pandas as pd

#Get Studio variables
if 'variables' in locals():
    if variables.get("SCORING_URI") is not None:
        scoring_uri = variables.get("SCORING_URI")
    if variables.get("SERVICE_KEY") is not None:
        service_key= variables.get("SERVICE_KEY")
    if variables.get("PREDICT_EXTENSION") is not None:
        predict_extension= variables.get("PREDICT_EXTENSION")
    if variables.get("LABEL_COLUMN") is not None:
        label= variables.get("LABEL_COLUMN")
        
API_ENDPOINT = variables.get("PREDICT_MODEL_ENDPOINT") if variables.get("PREDICT_MODEL_ENDPOINT") else variables.get("ENDPOINT_MODEL")
API_PREDICT_ENDPOINT = API_ENDPOINT+predict_extension
print("API_PREDICT_ENDPOINT",API_PREDICT_ENDPOINT)
service_token = variables.get("SERVICE_TOKEN") if variables.get("SERVICE_TOKEN") else variables.get("SERVICE_TOKEN_PROPAGATED")
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

#Get Data from previous tasks
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
  dataframe_test = dataframe.drop([label], axis=1, inplace=False)
  dataframe_json = dataframe_test.to_json(orient='values')
elif input_variables['task.dataframe_id'] is not None:
  dataframe_id = input_variables['task.dataframe_id']
  dataframe_json = variables.get(dataframe_id)
  dataframe_json = bz2.decompress(dataframe_json).decode()
  assert dataframe_json is not None
  dataframe = pd.read_json(dataframe_json, orient='split')
  dataframe_test = dataframe.drop([label], axis=1, inplace=False)
  dataframe_json = dataframe_test.to_json(orient='values')
elif variables.get("INPUT_DATA") is not None:
  dataframe_json = variables.get("INPUT_DATA")
else:
  print("there is no input data")

print ('dataframe_json',dataframe_json)

data = {'dataframe_json' : dataframe_json, 'api_token' : service_token}
data_json = json.dumps(data)
req = requests.post(API_PREDICT_ENDPOINT, data=data_json, headers=headers)

predictions = json.loads(req.text)
print("predictions:\n", predictions)

print("END " + __file__)