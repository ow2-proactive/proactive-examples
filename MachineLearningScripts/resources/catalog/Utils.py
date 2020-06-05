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


def preview_dataframe_in_task_result(dataframe):
    """
    Preview a Pandas dataframe as a Proactive task result.

    :param dataframe: Pandas dataframe.
    :return: None.
    """
    import pandas as pd
    global variables, result, resultMetadata
    limit_output_view = variables.get("LIMIT_OUTPUT_VIEW")
    limit_output_view = 5 if limit_output_view is None else int(limit_output_view)
    info = ''
    if limit_output_view > 0:
        info = "Task preview limited to " + str(limit_output_view) + " rows"
        print(info)
        dataframe = dataframe.head(limit_output_view).copy()
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
    <p align="center">{0}</p>
    <div style="text-align:center">{1}</div>
    </body></html>""".format(info, result)
    result = result.encode('utf-8')
    resultMetadata.put("file.extension", ".html")
    resultMetadata.put("file.name", "output.html")
    resultMetadata.put("content.type", "text/html")


def transfer_dataframe_in_dataspace(dataframe, dataspace="user"):
    """
    Transfer a Pandas dataframe to the user space.

    :param dataframe: Pandas dataframe.
    :param dataspace: Data space to be used [user, global]
    :return: None.
    """
    global variables, userspaceapi, globalspaceapi, gateway

    if dataspace is None:
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
