__file__ = variables.get("PA_TASK_NAME")
print("BEGIN " + __file__)

import requests
import json, bz2
import numpy as np
import pandas as pd

#Default variables initialization
headers = {'Content-Type':'application/json'}
scoring_uri = 'http://705492df-3699-40fd-b434-abb5ecb33c63.uksouth.azurecontainer.io/score'
service_key = ''

#Get Studio variables
if 'variables' in locals():
    if variables.get("SCORING_URI") is not None:
        scoring_uri = variables.get("SCORING_URI")
    if variables.get("SERVICE_KEY") is not None:
        service_key= variables.get("SERVICE_KEY")

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
  dataframe_test = dataframe.drop(['class'], axis=1, inplace=False)
  dataframe_json = dataframe_test.to_json(orient='values')
elif input_variables['task.dataframe_id'] is not None:
  dataframe_id = input_variables['task.dataframe_id']
  dataframe_json = variables.get(dataframe_id)
  dataframe_json = bz2.decompress(dataframe_json).decode()
  assert dataframe_json is not None
  dataframe = pd.read_json(dataframe_json, orient='split')
  dataframe_test = dataframe.drop(['class'], axis=1, inplace=False)
  dataframe_json = dataframe_test.to_json(orient='values')
elif variables.get("INPUT_DATA") is not None:
  dataframe_json = variables.get("INPUT_DATA")
else:
  print("there is no input data")

if service_key is not "":
    headers['Authorization'] = 'Bearer '+service_key

#Call the deployed service
response = requests.post(scoring_uri, data=dataframe_json, headers=headers)

#Print the service response
print(response.status_code)
print(response.elapsed)
print(response.text)

print("END " + __file__)