# -*- coding: utf-8 -*-
"""Proactive Python Utils for Machine Learning

Provide a collection of common utility Python functions and classes
for Proactive machine learning tasks and workflows.
"""
import os

global variables, result, results, resultMetadata
global userspaceapi, globalspaceapi, gateway


def raiser(msg=None):
    """
    Raise an AssertionError with the given message.

    :param msg: Default error message.
    :return: None.
    """
    raise AssertionError(msg or "Assertion error")


def assert_not_empty(s, msg=None):
    """
    Assert input string is not empty.

    :param s: Input string.
    :param msg: Default error message.
    :return: None.
    """
    if not s.strip():
        raiser(msg)


def assert_not_none(expr, msg=None):
    """
    Fail if the expression is None.

    :param expr: Tested expression.
    :param msg: Default error message.
    :return: None.
    """
    if expr is None:
        raiser(msg)


def assert_not_none_not_empty(s, msg=None):
    """
    Assert input string is not none and not empty.

    :param s: Input string.
    :param msg: Default error message.
    :return: None.
    """
    assert_not_none(s, msg)
    assert_not_empty(s, msg)


def assert_valid_int(s, msg=None):
    """
    Assert input string is a valid int.

    :param s: Input string.
    :param msg: Default error message.
    :return: Integer.
    """
    try:
        i = int(s)
    except ValueError:
        i = None
    assert_not_none(i, msg)
    return i


def assert_valid_float(s, msg=None):
    """
    Assert input string is a valid float.

    :param s: Input string.
    :param msg: Default error message.
    :return: Float.
    """
    try:
        f = float(s)
    except ValueError:
        f = None
    assert_not_none(f, msg)
    return f


def assert_greater_equal(first, second, msg=None):
    """
    Assert `first` is greater than or equal to second.

    :param first: First argument.
    :param second: Second argument.
    :param msg: Default error message.
    :return: None.
    """
    if not first >= second:
        if msg is None:
            msg = "{0} is not greater than or equal to {1}".format(first, second)
        raiser(msg)


def assert_less_equal(first, second, msg=None):
    """
    Assert `first` is less than or equal to second.

    :param first: First argument.
    :param second: Second argument.
    :param msg: Default error message.
    :return: None.
    """
    if not first <= second:
        if msg is None:
            msg = "{0} is not less than or equal to {1}".format(first, second)
        raiser(msg)


def assert_between(val, minvalue, maxvalue, msg=None):
    """
    Assert `val` is greater than or equal to `minvalue` and
    less than or equal to `maxvalue`.

    :param val: Number (int, float, double).
    :param minvalue: Number (int, float, double).
    :param maxvalue: Number (int, float, double).
    :param msg: Default error message.
    :return: None.
    """
    assert_greater_equal(val, minvalue, msg)
    assert_less_equal(val, maxvalue, msg)


def check_task_is_enabled():
    """
    Check if the task/workflow variable TASK_ENABLED exists,
    and if it's `false` then terminate the Python script.
    """
    # import sys
    global variables
    task_name = variables.get("PA_TASK_NAME")
    if str(variables.get("TASK_ENABLED")).lower() == 'false':
        print("Task " + task_name + " disabled")
        print("END " + task_name)
        quit()
        # sys.exit()


def get_input_variables(input_variables):
    """
    Get the propagated variables from Proactive `resultMetadata`.
    The input dictionary is updated with the corresponding values.

    :param input_variables: Python dictionary containing the variables name and its default value.
    >>> input_variables = {
    >>> 'task.dataframe_id': None,
    >>> 'task.label_column': None
    >>> }
    :return: None.
    """
    for key in input_variables.keys():
        for res in results:
            value = res.getMetadata().get(key)
            if value is not None:
                input_variables[key] = value
                break


def get_input_variables_from_key(input_variables, key):
    """
    Get the propagated variables from Proactive `resultMetadata`.
    The `input_variables` is updated with the corresponding values found via `key`.

    :param input_variables: Python dictionary containing the variables to be filled.
    >>> input_variables = {
    >>> 'dataframe_id1': None,
    >>> 'dataframe_id2': None
    >>> }
    :param key: Python string containing the key to be found.
    >>> key = 'task.dataframe_id'
    :return: None.
    """
    for res in results:
        value = res.getMetadata().get(key)
        if value is not None:
            for k in input_variables.keys():
                if input_variables[k] is None:
                    input_variables[k] = value
                    break


def preview_dataframe_in_task_result(dataframe, output_type=None):
    """
    Preview a Pandas dataframe in the following outputs type: HTML, CSV, and JSON.
    The DataFrame can be also exported to Tableau or S3.

    :param dataframe: Pandas dataframe.
    :param output_type: Python string: HTML, CSV, JSON, Tableau, S3 or None (default).
    :return: None.
    """
    output_list = ["html", "csv", "json", "tableau", "s3"]
    output_type = output_type.lower() if output_type is not None else output_type
    if output_type not in output_list:
        output_type = "html"

    if output_type == "html":
        export_dataframe_html(dataframe)

    if output_type == "s3":
        export_dataframe_s3(dataframe)

    if output_type == "csv":
        export_dataframe_csv(dataframe)

    if output_type == "json":
        export_dataframe_json(dataframe)

    if output_type == "tableau":
        export_dataframe_tableau(dataframe)


def export_dataframe_html(dataframe):
    """
    Export a Pandas dataframe as a HTML in the Proactive task result.

    :param dataframe: Pandas dataframe.
    :return: None.
    """
    global variables, result, resultMetadata
    import pandas as pd

    limit_output_view = variables.get("LIMIT_OUTPUT_VIEW")
    limit_output_view = 5 if limit_output_view is None else int(limit_output_view)
    if limit_output_view > 0:
        info = "The task preview is limited to " + str(limit_output_view) + " rows"
        dataframe = dataframe.head(limit_output_view).copy()
    else:
        nrows = len(dataframe.index)
        info = "Total of " + str(nrows) + " rows"
    print(info)
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
    <body>
    <h1 class="text-center my-4" style="color:#003050;">Data Preview</h1>
    <p align="center">{0}</p>
    {1}
    </body></html>""".format(info, result)
    result = result.encode('utf-8')
    resultMetadata.put("file.extension", ".html")
    resultMetadata.put("file.name", "output.html")
    resultMetadata.put("content.type", "text/html")


def export_dataframe_s3(dataframe):
    """
    Export a Pandas dataframe to the Amazon S3.

    :param dataframe: Pandas dataframe.
    :return: None.
    """
    global result
    import s3fs, uuid

    user_access_key_id = variables.get('UserAccessKeyID')
    user_secret_access_key = variables.get('UserSecretAccessKey')
    user_bucket_path = variables.get('UserBucketPath')

    assert_not_none_not_empty(user_access_key_id, "UserAccessKeyID should be defined!")
    assert_not_none_not_empty(user_secret_access_key, "UserSecretAccessKey should be defined!")

    dataframe_id = str(uuid.uuid4())
    print("dataframe id (out): ", dataframe_id)
    bytes_to_write = dataframe.to_csv(index=False).encode()

    fs = s3fs.S3FileSystem(
        key=str(user_access_key_id),
        secret=str(user_secret_access_key),
        s3_additional_kwargs={'ACL': 'public-read'}
    )

    bucket_path = str(user_bucket_path) if user_bucket_path is not None else 's3://activeeon-public/results/'
    print("Using the following bucket path: ", bucket_path)
    s3file_path = bucket_path + dataframe_id + '.csv'
    with fs.open(s3file_path, 'wb') as f:
        f.write(bytes_to_write)

    dataframe_url = fs.url(s3file_path).split('?')[0]
    dataframe_info = fs.info(s3file_path)
    print("The dataframe was uploaded successfully to the following url:")
    print(dataframe_url)
    print("File info:")
    print(dataframe_info)

    result = """
    <meta http-equiv="Refresh" content="0; url='{0}'" />
    """.format(dataframe_url)
    result = result.encode('utf-8')
    resultMetadata.put("file.extension", ".html")
    resultMetadata.put("file.name", "output.html")
    resultMetadata.put("content.type", "text/html")


def export_dataframe_csv(dataframe, filename="dataframe.csv"):
    """
    Export a Pandas dataframe to a CSV file.

    :param dataframe: Pandas dataframe.
    :param filename: Python string.
    :return: None.
    """
    global result
    dataframe.to_csv(filename, encoding='utf-8', index=False)
    with open(filename, "rb") as binary_file:
        file_bin = binary_file.read()
    assert file_bin is not None
    result = file_bin
    resultMetadata.put("file.extension", ".csv")
    resultMetadata.put("file.name", filename)
    resultMetadata.put("content.type", "text/csv")


def export_dataframe_json(dataframe, filename="dataframe.json"):
    """
    Export a Pandas dataframe to a JSON file.

    :param dataframe: Pandas dataframe.
    :param filename: Python string.
    :return: None.
    """
    global result
    dataframe.to_json(filename, orient='split')
    with open(filename, "rb") as binary_file:
        file_bin = binary_file.read()
    assert file_bin is not None
    result = file_bin
    resultMetadata.put("file.extension", ".json")
    resultMetadata.put("file.name", filename)
    resultMetadata.put("content.type", "application/json")


def export_dataframe_tableau(dataframe, filename="dataframe.hyper"):
    """
    Export a Pandas dataframe to a Tableau file.

    :param dataframe: Pandas dataframe.
    :param filename: Python string.
    :return: None.
    """
    global result
    import pantab
    # TODO: Fix ModuleNotFoundError: No module named 'tableauhyperapi'
    result = dataframe.to_json(orient='split')
    pantab.frame_to_hyper(dataframe, filename)
    with open(filename, "rb") as binary_file:
        file_bin = binary_file.read()
    assert file_bin is not None
    result = file_bin
    resultMetadata.put("file.extension", ".hyper")
    resultMetadata.put("file.name", filename)
    resultMetadata.put("content.type", "application/octet-stream")


def transfer_dataframe_in_dataspace(dataframe, dataspace="user"):
    """
    Transfer a Pandas dataframe to the user space.

    :param dataframe: Pandas dataframe.
    :param dataspace: Data space to be used [user, global]
    :return: None.
    """
    global variables, userspaceapi, globalspaceapi, gateway

    if dataspace is None or dataspace not in ["user", "global"]:
        dataspace = "user"

    job_id = variables.get("PA_JOB_ID")
    task_id = variables.get("PA_TASK_ID")

    dataframe_id = 'task_id_' + task_id
    dataframe_file_path = dataframe_id + '.csv.gz'
    dataframe.to_csv(dataframe_file_path, compression='gzip')
    destination_path = os.path.join('job_id_' + job_id, dataframe_file_path)

    print("Transferring dataframe to the " + dataspace + " space")
    print('File size in KB:  ', os.path.getsize(dataframe_file_path) / 1024)
    print('Destination path: ', destination_path)
    java_file = gateway.jvm.java.io.File(dataframe_file_path)

    # connect to the data space api
    if dataspace == "user":
        # $PA_SCHEDULER_HOME/data/defaultuser/
        userspaceapi.connect()
        userspaceapi.pushFile(java_file, destination_path)
    else:
        # $PA_SCHEDULER_HOME/data/defaultglobal/
        globalspaceapi.connect()
        globalspaceapi.pushFile(java_file, destination_path)


def compress_and_transfer_dataframe_in_variables(dataframe):
    """
    Compress and transfer a Pandas dataframe to the Proactive variables.

    :param dataframe: Pandas dataframe.
    :return: ID of the dataframe.
    """
    import uuid, bz2
    dataframe_json = dataframe.to_json(orient='split').encode()
    compressed_data = bz2.compress(dataframe_json)
    dataframe_id = str(uuid.uuid4())
    variables.put(dataframe_id, compressed_data)
    return dataframe_id


def get_and_decompress_dataframe(dataframe_id):
    """
    Get a Pandas dataframe from Proactive variables by using its ID.

    :param dataframe_id: Pandas dataframe id (uuid).
    :return: Pandas dataframe.
    """
    import bz2
    import pandas as pd
    dataframe_json = variables.get(dataframe_id)
    assert_not_none(dataframe_json, "Invalid dataframe!")
    dataframe_json = bz2.decompress(dataframe_json).decode()
    dataframe = pd.read_json(dataframe_json, orient='split')
    return dataframe


def encode_columns(dataframe, columns, sep=","):
    """
    Apply an encoder to columns of a Pandas dataframe.

    :param dataframe: Pandas dataframe.
    :param columns: String containing the columns name separated by comma.
    :param sep: Column separator. Default is comma ','.
    :return: Pandas dataframe and the encode map dictionary.
    """
    from sklearn.preprocessing import LabelEncoder
    if isinstance(columns, str):
        columns2encode = [x.strip() for x in columns.split(sep)]
    else:
        columns2encode = columns
    assert_not_none_not_empty(columns2encode, "The columns to be encoded should be defined!")
    encode_map = {}
    dataframe_aux = dataframe.copy()
    for col in columns2encode:
        unique_vector = dataframe[col].unique()
        label_encoder = LabelEncoder()
        label_encoder.fit(unique_vector)
        enc_values = label_encoder.transform(unique_vector)
        enc_map = dict(zip(unique_vector, enc_values))
        dataframe_aux[col] = dataframe[col].map(enc_map)
        encode_map[col] = enc_map
    return dataframe_aux, encode_map


def apply_encoder(dataframe, columns, encode_map, sep=","):
    """
    Apply an encode map to the specified columns of a Pandas dataframe.

    :param dataframe: Pandas dataframe.
    :param columns: String containing the columns name separated by comma.
    :param encode_map: Encode map dictionary.
    :param sep: Column separator. Default is comma ','.
    :return: Pandas dataframe.
    """
    if isinstance(columns, str):
        columns2encode = [x.strip() for x in columns.split(sep)]
    else:
        columns2encode = columns
    assert_not_none_not_empty(columns2encode, "The columns to be encoded should be defined!")
    dataframe_aux = dataframe.copy()
    for col in columns2encode:
        col_mapper = encode_map[col]
        dataframe_aux[col] = dataframe[col].map(col_mapper)
    return dataframe_aux
