__file__ = variables.get("PA_TASK_NAME")
print("BEGIN " + __file__)

import wget
from azureml.core.webservice import Webservice
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.model import InferenceConfig
from azureml.core.image import ContainerImage

#Define default variables
WORKSPACE_NAME = 'docs-ws'
SUBSCRIPTION_ID = 'a1c03dc2-0383-4ec9-9c73-fb74aa0de4f6'
RESOURCE_GROUP = 'test_azure_rs'
SERVICE_NAME = 'sklearn-mnist-test'
MODEL_PATH = "./myModel.pkl"
MODEL_NAME = "test_model"
CONDA_FILE_PATH = "./myenv.yml"
MODEL_DESCRIPTION = " test a new update of the service"
EXECUTION_SCRIPT_PATH = "./score.py"
model_id = None

#Get variables from the Studio
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
    if variables.get("EXECUTION_SCRIPT_URL") is not None:
        EXECUTION_SCRIPT_URL = variables.get("EXECUTION_SCRIPT_URL")
    if variables.get("CONDA_FILE_URL") is not None:
        CONDA_FILE_URL = variables.get("CONDA_FILE_URL")

#Set the interactive authentification
interactive_auth = InteractiveLoginAuthentication()

#Set the interactive authentification
ws = Workspace.get(name=AZURE_WORKSPACE_NAME, auth=interactive_auth, subscription_id=SUBSCRIPTION_ID,resource_group=RESOURCE_GROUP)
service = Webservice(workspace=ws, name=SERVICE_NAME)

#Get Model PATH
if 'variables' in locals():
    if variables.get(model_id) is not None:
        model_compressed = variables.get(model_id)
        model_bin = bz2.decompress(model_compressed)
        with bz2.open(MODEL_PATH, "wb") as f:
            model = f.write(model_bin)
        #pickle.dump(model_bin,open("sklearn_mnist.pkl","wb"))
        print('model size (original):   ', sys.getsizeof(MODEL_PATH), " bytes")
        MODEL_PATH = os.path.join(os.getcwd(),MODEL_PATH)
    else:
    	MODEL_PATH = os.path.join(os.getcwd(),MODEL_PATH)
    	wget.download(MODEL_URL,MODEL_PATH)

#Register a new model
new_model = Model.register(model_path = MODEL_PATH,
                       model_name = MODEL_NAME,
                       description = MODEL_DESCRIPTION,
                       workspace = ws)

#Create a new image
if CONDA_FILE_URL is not '':
    # Define the Conda environment
    from azureml.core.conda_dependencies import CondaDependencies
    myenv = CondaDependencies()
    myenv.add_conda_package("scikit-learn")
	#myenv.add_pip_package("joblib")
    with open("myenv.yml","w") as f:
        f.write(myenv.serialize_to_string())
    inference_config = InferenceConfig(entry_script=EXECUTION_SCRIPT_PATH, runtime="python", conda_file="myenv.yml")
else:
    wget.download(CONDA_FILE_URL,CONDA_FILE_PATH)
    inference_config = InferenceConfig(entry_script=EXECUTION_SCRIPT_PATH, runtime="python", conda_file=CONDA_FILE_PATH)

#Update the service
#service.update(image=None, tags=None, properties=None, description=None, auth_enabled=None, ssl_enabled=None, ssl_cert_pem_file=None, ssl_key_pem_file=None, ssl_cname=None, enable_app_insights=None, models=None, inference_config=None)
service.update(models=[new_model],inference_config=inference_config)
print(service.state)
print(service.get_logs())
print("service ",SERVICE_NAME," was updated successefuly")

variables.put("SCORING_URI",service.scoring_uri)

print("END " + __file__)