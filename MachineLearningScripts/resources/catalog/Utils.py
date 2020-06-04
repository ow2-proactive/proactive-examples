# -*- coding: utf-8 -*-
"""Proactive Python Utils for Machine Learning

Provide a collection of common utility Python functions and classes
for Proactive machine learning tasks and workflows.
"""
import os

global variables, result, resultMetadata
global userspaceapi, globalspaceapi, gateway


def raiser(msg): raise Exception(msg)


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
