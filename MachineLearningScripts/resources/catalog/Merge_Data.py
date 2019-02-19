__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import sys, bz2, uuid
import pandas as pd
import numpy as np

REF_COLUMN = variables.get("REF_COLUMN")
assert REF_COLUMN is not None and REF_COLUMN is not ""

input_variables = {'task.dataframe_id': None}
dataframe_id1 = None
dataframe_id2 = None

for key in input_variables.keys():
  for res in results:
    value = res.getMetadata().get(key)
    if value is not None and dataframe_id1 is None:
      dataframe_id1 = value
      continue
    if value is not None and dataframe_id2 is None:
      dataframe_id2 = value
      continue

print("dataframe id1 (in): ", dataframe_id1)
print("dataframe id2 (in): ", dataframe_id2)

dataframe_json1 = variables.get(dataframe_id1)
dataframe_json2 = variables.get(dataframe_id2)

assert dataframe_json1 is not None
assert dataframe_json2 is not None

dataframe_json1 = bz2.decompress(dataframe_json1).decode()
dataframe_json2 = bz2.decompress(dataframe_json2).decode()

dataframe1 = pd.read_json(dataframe_json1, orient='split')
dataframe2 = pd.read_json(dataframe_json2, orient='split')

dataframe = pd.merge(dataframe1, dataframe2, on=[REF_COLUMN], how='outer')

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

LIMIT_OUTPUT_VIEW = variables.get("LIMIT_OUTPUT_VIEW")
LIMIT_OUTPUT_VIEW = 5 if LIMIT_OUTPUT_VIEW is None else int(LIMIT_OUTPUT_VIEW)
if LIMIT_OUTPUT_VIEW > 0:
  print("task result limited to: ", LIMIT_OUTPUT_VIEW, " rows")
  dataframe = dataframe.head(LIMIT_OUTPUT_VIEW).copy()

#============================== Preview results ===============================
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
#==============================================================================

print("END " + __file__)