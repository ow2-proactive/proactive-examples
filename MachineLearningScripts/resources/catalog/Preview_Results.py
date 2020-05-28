__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() == 'false':
    print("Task " + __file__ + " disabled")
    quit()

print("BEGIN " + __file__)

import bz2

import pandas as pd

# -------------------------------------------------------------
# Get data from the propagated variables
#
OUTPUT_TYPE = variables.get("OUTPUT_TYPE")
assert OUTPUT_TYPE is not None and OUTPUT_TYPE is not ""
OUTPUT_TYPE = OUTPUT_TYPE.upper()

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

if OUTPUT_TYPE == "CSV":
    # result = dataframe.to_csv(encoding='utf-8', index=False)
    result = dataframe.to_csv(index=False)
    resultMetadata.put("file.extension", ".csv")
    resultMetadata.put("file.name", "dataframe.csv")
    resultMetadata.put("content.type", "text/csv")

if OUTPUT_TYPE == "JSON":
    result = dataframe.to_json(orient='split', encoding='utf-8')
    resultMetadata.put("file.extension", ".json")
    resultMetadata.put("file.name", "dataframe.json")
    resultMetadata.put("content.type", "application/json")

if OUTPUT_TYPE == "HTML":
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
            <h1 class="text-center my-4" style="color:#003050;">Machine Learning Results</h1>
            <div style="text-align:center">{0}</div>
            </body></html>
            """.format(result)
    result = result.encode('utf-8')
    resultMetadata.put("file.extension", ".html")
    resultMetadata.put("file.name", "output.html")
    resultMetadata.put("content.type", "text/html")

print("END " + __file__)
