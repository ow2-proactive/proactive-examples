__file__ = variables.get("PA_TASK_NAME")

print("BEGIN " + __file__)

import tableauserverclient as TSC
import pandas as pd
import pantab
import bz2

# Connection to the Tableau Server
# Create an account at https://www.tableau.com/products/server
# Add your credentials in the schedular portal
USERNAME = credentials.get("TABLEAU_SERVER_USERNAME")
PASSWORD = credentials.get("TABLEAU_SERVER_PASSWORD")

SERVER_ENDPOINT = variables.get("SERVER_ENDPOINT")
SITE_ID = variables.get("SITE_ID")
PROJECT_NAME = variables.get("PROJECT_NAME")
INPUT_FILE_NAME = variables.get("INPUT_FILE_NAME")

tableau_auth = TSC.TableauAuth(USERNAME, PASSWORD, site_id=SITE_ID)
server = TSC.Server(SERVER_ENDPOINT)
project_id = None

with server.auth.sign_in(tableau_auth):
    all_project_items, pagination_item = server.projects.get()
    # Check if the specified project_name exists
    for project in all_project_items:
        if project.name == PROJECT_NAME:
            project_id = project.id 
    if project_id is None:
        # Create a new project item
        new_project = TSC.ProjectItem(name=PROJECT_NAME, content_permissions='LockedToProject', id='LockedToProject')
        # Create the project 
        new_project = server.projects.create(new_project)
        # Create new datasource_item with project id '3a8b6148-493c-11e6-a621-6f3499394a39'

    # Save dataframe into a .hyper file
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
    pantab.frame_to_hyper(dataframe, INPUT_FILE_NAME)

    # Create new datasource_item
    new_datasource = TSC.DatasourceItem(project_id)
    new_datasource = server.datasources.publish(new_datasource, INPUT_FILE_NAME, 'CreateNew')
    print(new_datasource.__dict__)

print("END " + __file__)