"""Proactive Python Utils for Machine Learning

Provide a collection of common utility Python functions and classes
for Proactive machine learning tasks and workflows.
"""
import sys, os

global variables, result, results, resultMetadata, resultMap
global userspaceapi, globalspaceapi, gateway


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)


def install(package):
    """
    Install a Python package.

    :param package: Package name.
    :return: None.
    """
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def raiser_ex(msg): raise Exception(msg)


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
    if type(s) == str:
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


def assert_valid_string(s, msg=None):
    """
    Assert input 's' is a valid string.

    :param s: Input string.
    :param msg:  Default error message.
    :return: None.
    """
    if not type(s) == str:
        raiser(msg)


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


def is_not_empty(s):
    """
    Check input string is not empty.
    Returns True if not empty, and False if it's empty.

    :param s: Input string.
    :return: Boolean.
    """
    return bool(s.strip())


def is_not_none(expr):
    """
    Check if the expression is not None.

    :param expr: Tested expression.
    :return: Boolean.
    """
    return expr is not None


def is_not_none_not_empty(s):
    """
    Check if the input string is not none and not empty.

    :param s: Input string.
    :return: Boolean.
    """
    return is_not_none(s) and is_not_empty(s)


def str_to_bool(s, none_is_false=False):
    """
    Convert a string representation to True or False.

    :param s: String.
    :param none_is_false: Consider None as False (default=False).
    :return: Boolean.
    """
    if none_is_false and s is None:
        return False
    else:
        assert_valid_string(s, msg="The input should be a string type!")
        s = s.lower()
        if s in ('y', 'yes', 't', 'true', 'on', '1'):
            return True
        elif s in ('n', 'no', 'f', 'false', 'off', '0'):
            return False
        else:
            raise ValueError("Invalid truth value %r" % (s,))


def is_true(s, none_is_false=False):
    """
    Check if a string 's' is True or False.

    :param s: String.
    :param none_is_false: Consider None as False (default=False).
    :return: Boolean.
    """
    return str_to_bool(s, none_is_false)


def is_false(s, none_is_false=False):
    """
    Check if a string 's' is True or False.

    :param s: String.
    :param none_is_false: Consider None as False (default=False).
    :return: Boolean.
    """
    return not is_true(s, none_is_false)


def is_nvidia_rapids_enabled():
    nvidia_rapids_enabled = False
    if variables.get("USE_NVIDIA_RAPIDS") is not None:
        nvidia_rapids_enabled = str_to_bool(variables.get("USE_NVIDIA_RAPIDS"))
    if nvidia_rapids_enabled:
        try:
            import cudf
        except ImportError:
            print("NVIDIA RAPIDS is not available")
            nvidia_rapids_enabled = False
            pass
    return nvidia_rapids_enabled


def dict_to_obj(d):
    """
    Convert a Python dictionary to a Python object.

    :param d: Python dictionary.
    :return: Python object.
    """
    return obj(d)


def check_task_is_enabled():
    """
    Check if the task/workflow variable TASK_ENABLED exists,
    and if it's `false` then terminate the Python script.
    """
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

    :param dataframe: Pandas dataframe.
    :param output_type: Python string: HTML, CSV, JSON or None (default=HTML).
    :return: None.
    """
    output_list = ["html", "csv", "json"]
    output_type = output_type.lower() if output_type is not None else output_type
    if output_type not in output_list:
        output_type = "html"

    if output_type == "html":
        export_dataframe_html(dataframe)

    if output_type == "csv":
        export_dataframe_csv(dataframe)

    if output_type == "json":
        export_dataframe_json(dataframe)


def export_dataframe_in_task_result(dataframe, output_type=None):
    """
    Export a Pandas dataframe in the following outputs type: HTML, CSV, and JSON.
    The DataFrame can be also exported to Tableau or S3.

    :param dataframe: Pandas dataframe.
    :param output_type: Python string: HTML, CSV, JSON, Tableau, S3 or None (default=HTML).
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
    try:
        with pd.option_context('display.max_colwidth', -1):
            result = dataframe.to_html(escape=False, classes='table table-bordered table-striped', justify='center')
    except:
        with pd.option_context('display.max_colwidth', None):
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


def export_dataframe_json(dataframe, filename="dataframe.json", orient='split'):
    """
    Export a Pandas dataframe to a JSON file.

    :param dataframe: Pandas dataframe.
    :param filename: Python string.
    :param orient: Format of the JSON string.
    :return: None.
    """
    global result
    dataframe.to_json(filename, orient=orient)
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


def export_data_for_download(data, file_name, file_extension, content_type="application/octet-stream"):
    """
    Export data to be downloaded.

    :param data: Data.
    :param file_name: File name (string).
    :param file_extension: File extension (string).
    :param content_type: Content type (string).
    :return: None.
    """
    import pickle
    global result
    data_dumped = pickle.dumps(data)
    result = data_dumped
    resultMetadata.put("file.name", file_name)
    resultMetadata.put("file.extension", file_extension)
    resultMetadata.put("content.type", content_type)


def export_model_for_download(model, file_name="myModel.model"):
    """
    Export a machine learning model to be downloaded.

    :param model: Model object.
    :param file_name: Python string.
    :return: None.
    """
    assert model is not None
    export_data_for_download(model, file_name, ".model")


def export_file_to_dataspace(file_path, export_to=None):
    """
    Export file to the data space.

    :param file_path: File path.
    :param export_to: Define where data goes to (PA:USER_FILE/user, PA:GLOBAL_FILE/global).
    :return: Boolean.
    """
    from os.path import isfile, getsize

    global userspaceapi, globalspaceapi, gateway

    if export_to is None or export_to not in ["PA:USER_FILE", "user", "PA:GLOBAL_FILE", "global"]:
        export_to = "user"

    assert isfile(file_path)
    print('File size in KB: ', getsize(file_path) / 1024)
    java_file = gateway.jvm.java.io.File(file_path)

    if export_to.upper() == "PA:USER_FILE" or export_to.lower() == "user":
        # $PA_SCHEDULER_HOME/data/defaultuser/
        print("Transferring file to the user space")
        userspaceapi.connect()
        userspaceapi.pushFile(java_file, file_path)

    if export_to.upper() == "PA:GLOBAL_FILE" or export_to.lower() == "global":
        # $PA_SCHEDULER_HOME/data/defaultglobal/
        print("Transferring file to the global space")
        globalspaceapi.connect()
        globalspaceapi.pushFile(java_file, file_path)

    return True


def import_csv_file(file_path, file_delimiter, import_from):
    """
    Import a CSV file from the data space to a Pandas dataframe.

    :param file_path: CSV file path.
    :param file_delimiter: CSV file delimiter.
    :param import_from: Define where data comes from (PA:USER_FILE, PA:GLOBAL_FILE, PA:URL, PA:URI).
    :return: Pandas dataframe.
    """
    import pandas as pd

    if import_from.upper() == "PA:USER_FILE":
        print("Importing file from the user space")
        userspaceapi.connect()
        out_file = gateway.jvm.java.io.File(file_path)
        userspaceapi.pullFile(file_path, out_file)

    if import_from.upper() == "PA:GLOBAL_FILE":
        print("Importing file from the global space")
        globalspaceapi.connect()
        out_file = gateway.jvm.java.io.File(file_path)
        globalspaceapi.pullFile(file_path, out_file)

    dataframe = pd.read_csv(file_path, file_delimiter)
    return dataframe


def import_file_from_dataspace(file_path, import_from=None):
    """
    Import file from the data space.

    :param file_path: File path.
    :param import_from: Define where data comes from (PA:USER_FILE/user, PA:GLOBAL_FILE/global).
    :return: Boolean.
    """
    from os.path import isfile, getsize

    global userspaceapi, globalspaceapi, gateway

    if import_from is None or import_from not in ["PA:USER_FILE", "user", "PA:GLOBAL_FILE", "global"]:
        import_from = "user"

    out_file = gateway.jvm.java.io.File(file_path)

    if import_from.upper() == "PA:USER_FILE" or import_from.lower() == "user":
        print("Importing file from the user space")
        userspaceapi.connect()
        userspaceapi.pullFile(file_path, out_file)

    if import_from.upper() == "PA:GLOBAL_FILE" or import_from.lower() == "global":
        print("Importing file from the global space")
        globalspaceapi.connect()
        globalspaceapi.pullFile(file_path, out_file)

    assert isfile(file_path)
    print('File size in KB: ', getsize(file_path) / 1024)
    return True


def save_data(data, file_path):
    """
    Save data in a file path.

    :param data: Data (object/variable).
    :param file_path: File path (string).
    :return: None.
    """
    import pickle
    data_dumped = pickle.dumps(data)
    with open(file_path, "wb") as f:
        f.write(data_dumped)


def save_model(model, file_path="myModel.model"):
    """
    Save a machine learning model in a file path.

    :param model: Model (object).
    :param file_path: File path (string).
    :return: None.
    """
    save_data(model, file_path)


def compress_and_transfer_dataframe_in_dataspace(dataframe, orient, dataspace="user"):
    """
    Transfer a Pandas dataframe to the user space.

    :param dataframe: Pandas dataframe.
    :param orient: Format of the JSON string.
    :param dataspace: Data space to be used [user, global]
    :return: ID of the dataframe (a.k.a. dataspace file path).
    """
    import uuid, bz2

    global variables, userspaceapi, globalspaceapi, gateway

    if dataspace is None or dataspace not in ["user", "global"]:
        dataspace = "user"

    job_id = variables.get("PA_JOB_ID")
    task_id = variables.get("PA_TASK_ID")
    work_dir = variables.get("WORK_DIR") if variables.get("WORK_DIR") is not None else "."

    dataframe_file_name = 'task_id_' + task_id + '_df_out_' + str(uuid.uuid4())
    dataframe_file_path = dataframe_file_name + '.json.bz2'
    dataframe_json = dataframe.to_json(orient=orient).encode()
    with bz2.open(dataframe_file_path, 'w') as f:
        f.write(dataframe_json)
    destination_path = os.path.join(work_dir, 'job_id_' + job_id, dataframe_file_path)

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

    dataframe_id = destination_path
    return dataframe_id


def compress_and_transfer_dataframe_in_variables(dataframe, orient):
    """
    Compress and transfer a Pandas dataframe to the Proactive variables.

    :param dataframe: Pandas dataframe.
    :param orient: Format of the JSON string.
    :return: ID of the dataframe.
    """
    import uuid, bz2
    dataframe_json = dataframe.to_json(orient=orient).encode()
    compressed_data = bz2.compress(dataframe_json)
    dataframe_id = str(uuid.uuid4())
    variables.put(dataframe_id, compressed_data)
    print("dataframe id: ", dataframe_id)
    print('dataframe size (original):   ', sys.getsizeof(dataframe), " bytes")
    print('dataframe size (encoded):    ', sys.getsizeof(dataframe_json), " bytes")
    print('dataframe size (compressed): ', sys.getsizeof(compressed_data), " bytes")
    return dataframe_id


def compress_and_transfer_dataframe(dataframe, orient='split', mechanism=None):
    """
    Compress and transfer a Pandas dataframe to the selected mechanism.

    :param dataframe: Pandas dataframe.
    :param orient: Format of the JSON string.
    :param mechanism: Data transfer mechanism.
    :return: ID of the dataframe.
    """
    if mechanism is None or mechanism not in ["variables", "dataspace", "sharedfolder"]:
        mechanism = "dataspace"

    if mechanism == "variables":
        dataframe_id = compress_and_transfer_dataframe_in_variables(dataframe, orient)

    if mechanism == "dataspace":
        dataframe_id = compress_and_transfer_dataframe_in_dataspace(dataframe, orient)

    if mechanism == "sharedfolder":
        raise NotImplementedError("sharedfolder mechanism not implemented")

    return dataframe_id


def compress_and_transfer_data_in_variables(data, data_label="data"):
    """
    Compress and transfer data to the Proactive variables.

    :param data: Data (object/variable).
    :param data_label: Data label (string).
    :return: ID of the data (string).
    """
    import sys, bz2, uuid, pickle
    data_id = str(uuid.uuid4())
    data_dumped = pickle.dumps(data)
    data_compressed = bz2.compress(data_dumped)
    variables.put(data_id, data_compressed)
    print(data_label + ' id: ', data_id)
    print(data_label + ' size (original):   ', sys.getsizeof(data), " bytes")
    print(data_label + ' size (dumped):     ', sys.getsizeof(data_dumped), " bytes")
    print(data_label + ' size (compressed): ', sys.getsizeof(data_compressed), " bytes")
    return data_id


def compress_and_transfer_data(data, data_label="data", mechanism=None):
    """
    Compress and transfer data to the selected mechanism.

    :param data: Data (object/variable).
    :param data_label: Data label (string).
    :param mechanism: Data transfer mechanism.
    :return: ID of the data.
    """
    if mechanism is None or mechanism not in ["variables", "dataspace", "sharedfolder"]:
        mechanism = "variables"

    if mechanism == "variables":
        data_id = compress_and_transfer_data_in_variables(data, data_label)

    if mechanism == "dataspace":
        raise NotImplementedError("dataspace mechanism not implemented")

    if mechanism == "sharedfolder":
        raise NotImplementedError("sharedfolder mechanism not implemented")

    return data_id


def compress_and_transfer_model_in_variables(model):
    """
    Compress and transfer a machine learning model to the Proactive variables.

    :param model: Model object.
    :return: ID of the model.
    """
    model_id = compress_and_transfer_data_in_variables(model, "model")
    return model_id


def compress_and_transfer_model(model, mechanism=None):
    """
    Compress and transfer a machine learning model to the selected mechanism.

    :param model: Binary model.
    :param mechanism: Data transfer mechanism.
    :return: ID of the model.
    """
    model_id = compress_and_transfer_data(model, "model", mechanism)
    return model_id


def get_and_decompress_data_from_variables(data_id, data_label="data"):
    """
    Get data from the Proactive variables by using its ID.

    :param data_id: Data id (uuid).
    :param data_label: Data label (string).
    :return: Data (object/variable).
    """
    import sys, bz2, pickle
    data_compressed = variables.get(data_id)
    data_dumped = bz2.decompress(data_compressed)
    data = pickle.loads(data_dumped)
    print(data_label + ' id: ', data_id)
    print(data_label + ' size (original):   ', sys.getsizeof(data), " bytes")
    print(data_label + ' size (dumped):     ', sys.getsizeof(data_dumped), " bytes")
    print(data_label + ' size (compressed): ', sys.getsizeof(data_compressed), " bytes")
    return data


def get_and_decompress_data(data_id, data_label="data", mechanism=None):
    """
    Get data from Proactive by using its ID.

    :param data_id: Data id (uuid).
    :param data_label: Data label (string).
    :param mechanism: Data transfer mechanism.
    :return: Data (object/string).
    """
    data = None
    assert data_id is not None

    if mechanism is None or mechanism not in ["variables", "dataspace", "sharedfolder"]:
        mechanism = "variables"

    if mechanism == "variables":
        data = get_and_decompress_data_from_variables(data_id, data_label)

    if mechanism == "dataspace":
        raise NotImplementedError("dataspace mechanism not implemented")

    if mechanism == "sharedfolder":
        raise NotImplementedError("sharedfolder mechanism not implemented")

    return data


def get_and_decompress_model_from_variables(model_id):
    """
    Get a machine learning model from the Proactive variables by using its ID.

    :param model_id: Model id (uuid).
    :return: Model object.
    """
    model = get_and_decompress_data_from_variables(model_id, "model")
    return model


def get_and_decompress_model(model_id, mechanism=None):
    """
    Get a machine learning model from Proactive by using its ID.

    :param model_id: Model id (uuid).
    :param mechanism: Data transfer mechanism.
    :return: Binary model.
    """
    model = get_and_decompress_data(model_id, "model", mechanism)
    return model


def get_and_decompress_dataframe_from_variables(dataframe_id):
    """
    Get a Pandas dataframe in JSON format from Proactive variables by using its ID.

    :param dataframe_id: Pandas dataframe id (uuid).
    :return: JSON dataframe.
    """
    import bz2
    dataframe_json = variables.get(dataframe_id)
    assert_not_none(dataframe_json, "Invalid dataframe!")
    dataframe_json = bz2.decompress(dataframe_json).decode()
    return dataframe_json


def get_and_decompress_dataframe_from_dataspace(dataframe_id, dataspace=None):
    """
    Get a Pandas dataframe in JSON format from Proactive variables by using its ID.

    :param dataframe_id: Pandas dataframe id (a.k.a. dataspace file path).
    :param dataspace: Data space to be used [user, global]
    :return: JSON dataframe.
    """
    import uuid, bz2
    from os.path import isfile, getsize

    global variables, userspaceapi, globalspaceapi, gateway

    if dataspace is None or dataspace not in ["user", "global"]:
        dataspace = "user"

    task_id = variables.get("PA_TASK_ID")
    dataframe_file_name = 'task_id_' + task_id + '_df_in_' + str(uuid.uuid4())
    dataframe_file_path = dataframe_file_name + '.json.bz2'
    origin_path = dataframe_id

    print("Transferring dataframe from the " + dataspace + " space")
    print('Origin path: ', origin_path)
    java_file = gateway.jvm.java.io.File(dataframe_file_path)

    # connect to the data space api
    if dataspace == "user":
        # $PA_SCHEDULER_HOME/data/defaultuser/
        userspaceapi.connect()
        userspaceapi.pullFile(origin_path, java_file)
    else:
        # $PA_SCHEDULER_HOME/data/defaultglobal/
        globalspaceapi.connect()
        globalspaceapi.pullFile(origin_path, java_file)

    assert isfile(dataframe_file_path)
    print('File size in KB: ', getsize(dataframe_file_path) / 1024)
    with bz2.open(dataframe_file_path, 'rb') as f:
        uncompressed_data = f.read()
    dataframe_json = uncompressed_data.decode()
    return dataframe_json


def get_and_decompress_json_dataframe(dataframe_id, mechanism=None):
    """
    Get a Pandas dataframe from Proactive by using its ID.

    :param dataframe_id: Pandas dataframe id (uuid).
    :param mechanism: Data transfer mechanism.
    :return: Pandas dataframe.
    """
    dataframe_json = None

    if mechanism is None or mechanism not in ["variables", "dataspace", "sharedfolder"]:
        mechanism = "dataspace"

    if mechanism == "variables":
        dataframe_json = get_and_decompress_dataframe_from_variables(dataframe_id)

    if mechanism == "dataspace":
        dataframe_json = get_and_decompress_dataframe_from_dataspace(dataframe_id)

    if mechanism == "sharedfolder":
        raise NotImplementedError("sharedfolder mechanism not implemented")

    assert dataframe_json is not None
    return dataframe_json


def get_and_decompress_dataframe(dataframe_id, orient='split', mechanism=None):
    """
    Get a Pandas dataframe from Proactive by using its ID.

    :param dataframe_id: Pandas dataframe id (uuid).
    :param orient: Format of the JSON string.
    :param mechanism: Data transfer mechanism.
    :return: Pandas dataframe.
    """
    import pandas as pd
    dataframe_json = get_and_decompress_json_dataframe(dataframe_id, mechanism)
    dataframe = pd.read_json(dataframe_json, orient=orient)
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
    assert_not_none_not_empty(columns, "The columns to be encoded should be defined!")
    if type(columns) == str:
        columns2encode = [x.strip() for x in columns.split(sep)]
    else:
        columns2encode = columns
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
    assert_not_none_not_empty(columns, "The columns to be encoded should be defined!")
    if type(columns) == str:
        columns2encode = [x.strip() for x in columns.split(sep)]
    else:
        columns2encode = columns
    dataframe_aux = dataframe.copy()
    for col in columns2encode:
        col_mapper = encode_map[col]
        inv_col_mapper = {v: k for k, v in col_mapper.items()}
        dataframe_aux[col] = dataframe_aux[col].map(inv_col_mapper)
    return dataframe_aux


def scale_columns(dataframe, columns, scaler_name="RobustScaler"):
    """
    Apply a data normalization method to the specified columns of a Pandas dataframe.

    :param dataframe: Pandas dataframe.
    :param columns: String containing the columns name separated by comma.
    :param scaler_name: String containing the name of the scaler method (default="RobustScaler").
    :return: Pandas dataframe and the scaler object.
    """
    import pandas as pd
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

    data = dataframe.filter(columns, axis=1)
    print(scaler.fit(data))

    scaled_data = scaler.transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=columns)

    dataframe_scaled = dataframe.copy()
    dataframe_scaled = dataframe_scaled.reset_index(drop=True)
    for column in columns:
        dataframe_scaled[column] = scaled_df[column]

    return dataframe_scaled, scaler


def apply_scaler(dataframe, columns, scaler):
    """
    Apply a scaler object to the specified columns of a Pandas dataframe.

    :param dataframe: Pandas dataframe.
    :param columns: String containing the columns name separated by comma.
    :param scaler: Scaler object.
    :return: Pandas dataframe.
    """
    import pandas as pd

    data = dataframe.filter(columns, axis=1)
    scaled_data = scaler.transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=columns)

    dataframe_scaled = dataframe.copy()
    dataframe_scaled = dataframe_scaled.reset_index(drop=True)
    for column in columns:
        dataframe_scaled[column] = scaled_df[column]

    return dataframe_scaled


def compute_global_model(dataframe, columns, bins, model_type="KMeans"):
    """
    Compute a global model from a Pandas dataframe.

    :param dataframe: Pandas dataframe.
    :param columns: List of the columns name.
    :param bins: Number of bins.
    :param model_type: Type of the model (KMeans, PolynomialFeatures).
    :return: Model.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import PolynomialFeatures

    models = {}
    for j, column in enumerate(columns):
        column_df = dataframe[column]
        X = column_df.values

        if model_type == "KMeans":
            model = KMeans(n_clusters=bins[j], random_state=0).fit(X.reshape(-1, 1))

        if model_type == "PolynomialFeatures":
            model = PolynomialFeatures().fit(X.reshape(-1, 1))

        models[column] = model
    return models


def compute_features(dataframe, columns, bins, model, model_type="KMeans"):
    """
    Compute the features of the specified columns from a Pandas dataframe using the given model.

    :param dataframe: Pandas dataframe.
    :param columns: List of the columns name.
    :param bins: Number of bins.
    :param model: Model.
    :param model_type: Type of the model.
    :return: Features.
    """
    import numpy as np
    import scipy.stats.stats as st
    row = []
    for j, column in enumerate(columns):
        column_df = dataframe[column]
        X = column_df.values

        if model is not None:
            if model_type == "KMeans":
                r = model[column].predict(X.reshape(-1, 1))

            if model_type == "PolynomialFeatures":
                r = model[column].transform(X.reshape(-1, 1)).tolist()
        else:
            r = X

        # compute feature histogram
        # counts, bin_edges = np.histogram(result, bins=bins[j], density=False)
        # column_hist = counts

        # compute normalized feature histogram
        counts, bin_edges = np.histogram(r, bins=bins[j], density=True)
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


def get_summary(dataframe, columns, bins, model, model_type, ref_column, label_column=None):
    """
    Get a summary of a Pandas dataframe using the given model.

    :param dataframe: Pandas dataframe.
    :param columns: List of the columns name.
    :param bins: Number of bins.
    :param model: Model.
    :param model_type: Type of the model.
    :param ref_column: Reference column.
    :param label_column: Label column (optional).
    :return: Pandas dataframe.
    """
    data = {}
    IDs = dataframe[ref_column].unique()
    for i, ID in enumerate(IDs):
        sub_df = dataframe.loc[dataframe[ref_column] == ID]
        row = [ID]

        features = compute_features(sub_df, columns, bins, model, model_type)
        row.extend(features)

        if label_column is not None and label_column is not "":
            assert sub_df[label_column].unique().shape[0] == 1
            label = sub_df[label_column].unique()[0]
            row.extend([label])

        data[i] = row
    return data
