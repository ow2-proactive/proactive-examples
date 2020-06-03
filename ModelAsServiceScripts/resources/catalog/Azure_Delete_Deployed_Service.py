__file__ = variables.get("PA_TASK_NAME")
print("BEGIN " + __file__)

from azureml.core.webservice import Webservice
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace

#Define Default values
WORKSPACE_NAME = 'docs-ws'
SUBSCRIPTION_ID = 'a1c03dc2-0383-4ec9-9c73-fb74aa0de4f6'
RESOURCE_GROUP = 'test_azure_rs'
SERVICE_NAME = 'sklearn-mnist-test'

#Get variables from the Studio
if 'variables' in locals():
    if variables.get("AZURE_SUBSCRIPTION_ID") is not None:
        AZURE_SUBSCRIPTION_ID = variables.get("AZURE_SUBSCRIPTION_ID")
    if variables.get("AZURE_RESOURCE_GROUP") is not None:
        AZURE_RESOURCE_GROUP = variables.get("AZURE_RESOURCE_GROUP")
    if variables.get("AZURE_WORKSPACE_NAME") is not None:
        AZURE_WORKSPACE_NAME = variables.get("AZURE_WORKSPACE_NAME")
    if variables.get("SERVICE_NAME") is not None:
        SERVICE_NAME = variables.get("SERVICE_NAME")

#Set the interactive authetification
interactive_auth = InteractiveLoginAuthentication()

#Access to the chosen workspace
ws = Workspace.get(name=AZURE_WORKSPACE_NAME, auth=interactive_auth, subscription_id=AZURE_SUBSCRIPTION_ID,resource_group=AZURE_RESOURCE_GROUP)

#Delete the chosen service
service = Webservice(workspace=ws, name=SERVICE_NAME)
service.delete()

print("service ",SERVICE_NAME,"was deleted successefuly")

print("END " + __file__)