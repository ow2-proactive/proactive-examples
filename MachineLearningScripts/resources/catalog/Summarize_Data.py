__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() == 'false':
    print("Task " + __file__ + " disabled")
    quit()

print("BEGIN " + __file__)

import bz2
import sys
import uuid

import numpy as np
import pandas as pd


def compute_global_model(df, columns, bins, model_type="KMeans"):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import PolynomialFeatures

    models = {}
    for j, column in enumerate(columns):
        column_df = df[column]
        X = column_df.values

        if model_type == "KMeans":
            model = KMeans(n_clusters=bins[j], random_state=0).fit(X.reshape(-1, 1))

        if model_type == "PolynomialFeatures":
            model = PolynomialFeatures().fit(X.reshape(-1, 1))

        models[column] = model
    return models


def compute_features(sub_df, columns, bins, model, model_type="KMeans"):
    import scipy.stats.stats as st
    row = []
    for j, column in enumerate(columns):
        column_df = sub_df[column]
        X = column_df.values

        if model is not None:
            if model_type == "KMeans":
                result = model[column].predict(X.reshape(-1, 1))

            if model_type == "PolynomialFeatures":
                result = model[column].transform(X.reshape(-1, 1)).tolist()
        else:
            result = X

        # compute feature histogram
        # counts, bin_edges = np.histogram(result, bins=bins[j], density=False)
        # column_hist = counts

        # compute normalized feature histogram
        counts, bin_edges = np.histogram(result, bins=bins[j], density=True)
        column_hist = counts * np.diff(bin_edges)

        row.extend(column_hist)

        # add extra features
        kurtosis = st.kurtosis(X.reshape(-1, 1))[0]
        skew = st.skew(X.reshape(-1, 1))[0]
        min_value = column_df.min()
        max_value = column_df.max()
        mean_value = column_df.mean()
        median_value = column_df.median()
        row.extend([kurtosis, skew, min_value, max_value, mean_value, median_value])
    return row


def get_summary(df, columns, bins, model, model_type, ref_column, label_column=None):
    data = {}
    IDs = df[ref_column].unique()
    for i, ID in enumerate(IDs):
        sub_df = df.loc[df[ref_column] == ID]
        row = [ID]

        features = compute_features(sub_df, columns, bins, model, model_type)
        row.extend(features)

        if label_column is not None and label_column is not "":
            assert sub_df[label_column].unique().shape[0] == 1
            label = sub_df[label_column].unique()[0]
            row.extend([label])

        data[i] = row
    return data


# -------------------------------------------------------------
# Get data from the propagated variables
#
REF_COLUMN = variables.get("REF_COLUMN")
assert REF_COLUMN is not None and REF_COLUMN is not ""
IGNORE_COLUMNS = [REF_COLUMN]

LABEL_COLUMN = variables.get("LABEL_COLUMN")
if LABEL_COLUMN is not None and LABEL_COLUMN is not "":
    IGNORE_COLUMNS.append(LABEL_COLUMN)

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

columns = dataframe.drop(IGNORE_COLUMNS, axis=1, inplace=False).columns.values
ncolumns = columns.shape[0]
bins = [10] * ncolumns
print(columns, ncolumns, bins)

model = None
GLOBAL_MODEL_TYPE = variables.get("GLOBAL_MODEL_TYPE")
if GLOBAL_MODEL_TYPE is not None and GLOBAL_MODEL_TYPE is not "":
    print('Computing the global model using ', GLOBAL_MODEL_TYPE)
    model = compute_global_model(dataframe, columns, bins, GLOBAL_MODEL_TYPE)
    print('Finished')

print('Summarizing data...')
data = get_summary(dataframe, columns, bins, model, GLOBAL_MODEL_TYPE, REF_COLUMN, LABEL_COLUMN)
print('Finished')

dataframe = pd.DataFrame.from_dict(data, orient='index')
cols_len = len(dataframe.columns)
dataframe.columns = list(range(0, cols_len))

COLUMNS_NAME = {0: REF_COLUMN}
if LABEL_COLUMN is not None and LABEL_COLUMN is not "":
    COLUMNS_NAME = {0: REF_COLUMN, cols_len - 1: LABEL_COLUMN}
dataframe.rename(index=str, columns=COLUMNS_NAME, inplace=True)

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
# -------------------------------------------------------------

print("END " + __file__)
