__file__ = variables.get("PA_TASK_NAME")
print("BEGIN " + __file__)

import numpy as np
import wget
import matplotlib.pyplot as plt
import azureml.core
import os, sys, bz2
import pickle
from sklearn.externals import joblib
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage

#Define the values by default
AZURE_SUBSCRIPTION_ID = 'a1c03dc2-0383-4ec9-9c73-fb74aa0de4f6'
AZURE_RESOURCE_GROUP = 'test_azure_rs'
AZURE_WORKSPACE_NAME = 'docs-ws'
MODEL_NAME = 'mymodel'
MODEL_PATH = 'sklearn_mnist_model.pkl'
MODEL_DESCRIPTION = 'digit classification'
SERVICE_NAME = 'sklearn-mnist-test'
SERVICE_DESCRIPTION = 'Predict MNIST with sklearn'
MEMORY_GB = 1
CPU_CORES = 1
EXECUTION_SCRIPT_PATH = "./score.py"
CONDA_FILE_PATH = "./myenv.yml"
DOCKER_FILE_PATH = "./dockerfile"
AUTH_ENABLED = None

#Get the variables from the Studio
if 'variables' in locals():
    if variables.get("AZURE_SUBSCRIPTION_ID") is not None:
        AZURE_SUBSCRIPTION_ID = variables.get("AZURE_SUBSCRIPTION_ID")
    if variables.get("AZURE_RESOURCE_GROUP") is not None:
        AZURE_RESOURCE_GROUP = variables.get("AZURE_RESOURCE_GROUP")
    if variables.get("AZURE_WORKSPACE_NAME") is not None:
        AZURE_WORKSPACE_NAME = variables.get("AZURE_WORKSPACE_NAME")
    if variables.get("MODEL_NAME") is not None:
        MODEL_NAME = variables.get("MODEL_NAME")
    if variables.get("MODEL_URL") is not None:
        MODEL_URL = variables.get("MODEL_URL")
    if variables.get("MODEL_DESCRIPTION") is not None:
        MODEL_DESCRIPTION = variables.get("MODEL_DESCRIPTION")
    if variables.get("SERVICE_NAME") is not None:
        SERVICE_NAME = variables.get("SERVICE_NAME")
    if variables.get("SERVICE_DESCRIPTION") is not None:
        SERVICE_DESCRIPTION = variables.get("SERVICE_DESCRIPTION")
    if variables.get("MEMORY_GB") is not None:
        MEMORY_GB = int(variables.get("MEMORY_GB"))
    if variables.get("CPU_CORES") is not None:
        CPU_CORES = int(variables.get("CPU_CORES"))
    if variables.get("EXECUTION_SCRIPT_URL") is not None:
        EXECUTION_SCRIPT_URL = variables.get("EXECUTION_SCRIPT_URL")
        wget.download(EXECUTION_SCRIPT_URL,EXECUTION_SCRIPT_PATH)
    if variables.get("CONDA_FILE_URL") is not None:
        CONDA_FILE_URL = variables.get("CONDA_FILE_URL")
    if variables.get("DOCKER_FILE_URL") is not None:
        DOCKER_FILE_URL = variables.get("DOCKER_FILE_URL")
    if variables.get("AUTH_ENABLED") is not None:
        AUTH_ENABLED = variables.get("AUTH_ENABLED")
        
input_variables = {'task.model_id': None}
for key in input_variables.keys():
  for res in results:
    value = res.getMetadata().get(key)
    if value is not None:
      input_variables[key] = value
      break

model_id = input_variables['task.model_id']


if 'variables' in locals():
    if variables.get(model_id) is not None:
        model_compressed = variables.get(model_id)
        model_bin = bz2.decompress(model_compressed)
        with open(MODEL_PATH, "wb") as f:
            model = f.write(model_bin)
        #pickle.dump(model_bin,open(MODEL_PATH,"wb"))
        print('model size (original):   ', sys.getsizeof(MODEL_PATH), " bytes")
        MODEL_PATH = os.path.join(os.getcwd(),MODEL_PATH)
    else:
    	MODEL_PATH = os.path.join(os.getcwd(),MODEL_PATH)
    	wget.download(MODEL_URL,MODEL_PATH)

#Set the inetractive authentication
interactive_auth = InteractiveLoginAuthentication()

#Get the chosen workspace
ws = Workspace.get(name=AZURE_WORKSPACE_NAME, auth=interactive_auth, subscription_id=AZURE_SUBSCRIPTION_ID,resource_group=AZURE_RESOURCE_GROUP)

print(ws.name, ws.location, ws.resource_group, ws.location, sep = '\t')

# register model
model = Model.register(model_name=MODEL_NAME, model_path=MODEL_PATH, workspace=ws, description = MODEL_DESCRIPTION)
print(model.name, model.id, model.version, sep = '\t')

#Set the image
aciconfig = AciWebservice.deploy_configuration(cpu_cores=CPU_CORES, memory_gb=MEMORY_GB, description=SERVICE_DESCRIPTION, auth_enabled=AUTH_ENABLED)

if CONDA_FILE_URL=='' and DOCKER_FILE_URL=='':
    from azureml.core.conda_dependencies import CondaDependencies
    myenv = CondaDependencies()
    myenv.add_conda_package("scikit-learn")
	#myenv.add_pip_package("joblib")
    with open("myenv.yml","w") as f:
        f.write(myenv.serialize_to_string())
    # configure the image
    image_config = ContainerImage.image_configuration(execution_script=EXECUTION_SCRIPT_PATH, runtime="python", conda_file="myenv.yml")
elif CONDA_FILE_URL is not '' and DOCKER_FILE_URL=='':
    wget.download(CONDA_FILE_URL,CONDA_FILE_PATH)
    image_config = ContainerImage.image_configuration(execution_script=EXECUTION_SCRIPT_PATH, runtime="python", conda_file=CONDA_FILE_PATH)
elif DOCKER_FILE_URL is not '':
    wget.download(DOCKER_FILE_URL,DOCKER_FILE_PATH)
    image_config = ContainerImage.image_configuration(execution_script=EXECUTION_SCRIPT_PATH, runtime="python", docker_file=DOCKER_FILE_PATH)

#Deploy the service
service = Webservice.deploy_from_model(workspace=ws, name=SERVICE_NAME, deployment_config=aciconfig, models=[model], image_config=image_config)

service.wait_for_deployment(show_output=True)

print("SERVICE_ENDPOINT",service.scoring_uri)

variables.put("SCORING_URI",service.scoring_uri)

if AUTH_ENABLED:
    primary, secondary = service.get_keys()
    variables.put("SERVICE_KEY",primary)

print("END " + __file__)