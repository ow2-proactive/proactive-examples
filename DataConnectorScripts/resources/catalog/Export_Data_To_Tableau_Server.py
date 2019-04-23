__file__ = variables.get("PA_TASK_NAME")

print("BEGIN " + __file__)

import tableauserverclient as TSC
import pandas as pd
import pantab
import bz2

#connexion to tableau server
#Add your credentials in the schedular portal
USERNAME=credentials.get("TABLEAU_SERVER_USERNAME")
PASSWORD=credentials.get("TABLEAU_SERVER_PASSWORD")

SERVER_ENDPOINT = variables.get("SERVER_ENDPOINT")
SITE_ID = variables.get("SITE_ID")
PROJECT_NAME = variables.get("PROJECT_NAME")
OUTPUT_FILE_NAME = variables.get("OUTPUT_FILE_NAME")

tableau_auth = TSC.TableauAuth(USERNAME, PASSWORD, site_id=SITE_ID)
#tableau_auth = TSC.TableauAuth('lolyne.pacheco@gmail.com', 'proactive123', site_id='carolinetableautest')
server = TSC.Server(SERVER_ENDPOINT)
project_id = None

with server.auth.sign_in(tableau_auth):
    all_project_items, pagination_item = server.projects.get()
    #check if the specified project_name exists
    for project in all_project_items:
        if project.name == PROJECT_NAME:
            project_id=project.id 
    if project_id is None:
        # create a new project item
    	new_project = TSC.ProjectItem(name=PROJECT_NAME, content_permissions='LockedToProject', id='LockedToProject')
    	# create the project 
    	new_project = server.projects.create(new_project)
    	# Create new datasource_item with project id '3a8b6148-493c-11e6-a621-6f3499394a39'
        
    #save dataframe into a .hyper file
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
    output_file = OUTPUT_FILE_NAME
    pantab.frame_to_hyper(dataframe, output_file)
    
    # Create new datasource_item
    new_datasource = TSC.DatasourceItem(project_id)
    new_datasource = server.datasources.publish(new_datasource, output_file, 'CreateNew')
    print(new_datasource.__dict__)

print("END " + __file__)