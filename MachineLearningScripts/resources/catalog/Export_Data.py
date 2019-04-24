__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import pandas as pd
import numpy as np
import bz2
import pantab

OUTPUT_TYPE = variables.get("OUTPUT_TYPE")
assert OUTPUT_TYPE is not None and OUTPUT_TYPE is not ""

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
print(dataframe.head())

OUTPUT_TYPE = OUTPUT_TYPE.upper()
if OUTPUT_TYPE == "S3":
  import s3fs, uuid

  UserAccessKeyID=str(variables.get('UserAccessKeyID'))
  UserSecretAccessKey=str(variables.get('UserSecretAccessKey'))
  UserBucketPath=variables.get('UserBucketPath')

  dataframe_id = str(uuid.uuid4())
  print("dataframe id (out): ", dataframe_id)
  bytes_to_write = dataframe.to_csv(index=False).encode()

  fs = s3fs.S3FileSystem(
      key=UserAccessKeyID, 
      secret=UserSecretAccessKey,
      s3_additional_kwargs={'ACL': 'public-read'}
  )

  bucket_path=str(UserBucketPath) if UserBucketPath is not None else 's3://activeeon-public/results/'
  s3file_path = bucket_path+dataframe_id+'.csv'
  with fs.open(s3file_path, 'wb') as f:
    f.write(bytes_to_write)

  dataframe_url = fs.url(s3file_path).split('?')[0]
  dataframe_info = fs.info(s3file_path)
  print("The dataframe was uploaded successfully to the following url:")
  print(dataframe_url)
  print("File info:")
  print(dataframe_info)

if OUTPUT_TYPE == "CSV":
  #result = dataframe.to_csv(encoding='utf-8', index=False)
  result = dataframe.to_csv(index=False)
  resultMetadata.put("file.extension", ".csv")
  resultMetadata.put("file.name", "dataframe.csv")
  resultMetadata.put("content.type", "text/csv")

if OUTPUT_TYPE == "JSON":
  result = dataframe.to_json(orient='split', encoding='utf-8')
  resultMetadata.put("file.extension", ".json")
  resultMetadata.put("file.name", "dataframe.json")
  resultMetadata.put("content.type", "application/json")
    
if OUTPUT_TYPE == "TABLEAU":
  result = dataframe.to_json(orient='split')
  pantab.frame_to_hyper(dataframe, "dataframe.hyper")
  FILE_BIN = None
  with open("dataframe.hyper", "rb") as binary_file:
      FILE_BIN = binary_file.read()
  assert FILE_BIN is not None  
  result = FILE_BIN
  resultMetadata.put("file.extension", ".hyper")
  resultMetadata.put("file.name", "dataframe.hyper")
  resultMetadata.put("content.type", "application/octet-stream") 

if OUTPUT_TYPE == "HTML":
  LIMIT_OUTPUT_VIEW = variables.get("LIMIT_OUTPUT_VIEW")
  LIMIT_OUTPUT_VIEW = 5 if LIMIT_OUTPUT_VIEW is None else int(LIMIT_OUTPUT_VIEW)
  if LIMIT_OUTPUT_VIEW > 0:
    print("task result limited to: ", LIMIT_OUTPUT_VIEW, " rows")
    dataframe = dataframe.head(LIMIT_OUTPUT_VIEW).copy()
  
  #***************# HTML PREVIEW STYLING #***************#
  styles = [
    dict(selector="th", props=[("font-weight", "bold"),
                               ("text-align", "center"),
                               ("font-size", "15px"),
                               ("background", "#0B6FA4"),
                               ("color", "#FFFFFF")]),
                               ("padding", "3px 7px"),
    dict(selector="td", props=[("text-align", "right"),
                               ("padding", "3px 3px"),
                               ("border", "1px solid #999999"),
                               ("font-size", "13px"),
                               ("border-bottom", "1px solid #0B6FA4")]),
    dict(selector="table", props=[("border", "1px solid #999999"),
                               ("text-align", "center"),
                               ("width", "100%"),
                               ("border-collapse", "collapse")])
  ]
  #******************************************************#

  with pd.option_context('display.max_colwidth', -1):
    result = dataframe.style.set_table_styles(styles).render().encode('utf-8')
    resultMetadata.put("file.extension", ".html")
    resultMetadata.put("file.name", "output.html")
    resultMetadata.put("content.type", "text/html")

print("END " + __file__)