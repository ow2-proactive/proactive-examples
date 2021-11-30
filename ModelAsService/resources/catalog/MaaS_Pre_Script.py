import ssl
import subprocess
import sys
import json
from cryptography.fernet import Fernet

global schedulerapi, variables

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Import required Python libraries according to Python version
try:
    from urllib.request import Request, urlopen  # Python 3
except ImportError:
    from urllib2 import Request, urlopen  # Python 2

try:
    import cryptography
except ImportError:
    install('cryptography')
    import cryptography

# Get user credentials and convert to json
schedulerapi.connect()
sessionId = str(schedulerapi.getSession())
connectionInfo = schedulerapi.getConnectionInfo()
ciLogin = str(connectionInfo.getLogin())
ciPasswd = str(connectionInfo.getPassword())
ciUrl = str(connectionInfo.getUrl())
user_credentials = {
  'sessionId': sessionId,
  'ciLogin': ciLogin,
  'ciPasswd': ciPasswd,
  'ciUrl': ciUrl
}
user_credentials_json = json.dumps(user_credentials)

# Encrypt user data into a binary file
key = Fernet.generate_key()
f = Fernet(key)
message = user_credentials_json.encode()
encrypted = f.encrypt(message)
user_data_file = 'user_data.enc'
with open(user_data_file, 'wb') as f:
    f.write(encrypted)
variables.put("USER_KEY", key.decode())
variables.put("USER_DATA_FILE", user_data_file)

# Get workflows variables
PA_CATALOG_REST_URL = variables.get("PA_CATALOG_REST_URL")
PYTHON_ENTRYPOINT = variables.get("PYTHON_ENTRYPOINT")
YAML_FILE = variables.get("YAML_FILE")

PA_MAAS_RESOURCES_URL = "/buckets/model-as-a-service/resources/"
python_file_url = PA_CATALOG_REST_URL + PA_MAAS_RESOURCES_URL + PYTHON_ENTRYPOINT + "/raw"
yaml_file_url   = PA_CATALOG_REST_URL + PA_MAAS_RESOURCES_URL + YAML_FILE + "/raw"
print("python_file_url: ", python_file_url)
print("yaml_file_url:   ", yaml_file_url)

# Download the two configuration file "ml_service" for the service definition
req_py = Request(python_file_url)
req_py.add_header('sessionid', sessionId)
if python_file_url.startswith('https'):
    context = ssl._create_unverified_context()
    python_file = urlopen(req_py, context=context).read()
else:
    python_file = urlopen(req_py).read()
python_content = python_file.decode('utf-8')
python_file_name = PYTHON_ENTRYPOINT + ".py"
with open(python_file_name, 'w') as f:
    f.write(python_content)

# Download the configuration file "ml_service-api" for the swagger specification
req_yaml = Request(yaml_file_url)
req_yaml.add_header('sessionid', sessionId)
if yaml_file_url.startswith('https'):
    context = ssl._create_unverified_context()
    yaml_file = urlopen(req_yaml, context=context).read()
else:
    yaml_file = urlopen(req_yaml).read()
yaml_file_content = yaml_file.decode('utf-8')
yaml_file_name = YAML_FILE + ".yaml"
with open(yaml_file_name, 'w') as f:
    f.write(yaml_file_content)