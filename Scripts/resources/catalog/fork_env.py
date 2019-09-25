#%% fork environment (python)
import os

# check if DOCKER_ENABLED variable is set 
DOCKER_ENABLED = True
if variables.get("DOCKER_ENABLED") is not None:
    if str(variables.get("DOCKER_ENABLED")).lower() == 'false':
        DOCKER_ENABLED = False

# check if the current environment is already in a docker container
if os.path.isfile('/.dockerenv'):
    DOCKER_ENABLED = False

# check if CUDA is enabled
CUDA_ENABLED = False
CUDA_HOME = os.getenv('CUDA_HOME', None)
CUDA_HOME_DEFAULT = '/usr/local/cuda'
if CUDA_HOME is not None:
    if os.path.isdir(CUDA_HOME) == True:
        CUDA_ENABLED = True
else:
    if os.path.isdir(CUDA_HOME_DEFAULT) == True:
        CUDA_ENABLED = True

# check if DOCKER_GPU_ENABLED variable is set
DOCKER_GPU_ENABLED = True
if variables.get("DOCKER_GPU_ENABLED") is not None:
    if str(variables.get("DOCKER_GPU_ENABLED")).lower() == 'false':
        DOCKER_GPU_ENABLED = False
if not CUDA_ENABLED:
    DOCKER_GPU_ENABLED = False

# check if user wants to use NVIDIA RAPIDS
USE_NVIDIA_RAPIDS = False
if variables.get("USE_NVIDIA_RAPIDS") is not None:
    if str(variables.get("USE_NVIDIA_RAPIDS")).lower() == 'true':
        USE_NVIDIA_RAPIDS = True

# set the default docker environment
DEFAULT_DOCKER_IMAGE = 'activeeon/dlm3'
DOCKER_RUN_CMD = 'docker run '

# activate CUDA support if DOCKER_GPU_ENABLED is True
if DOCKER_GPU_ENABLED:
    DOCKER_RUN_CMD += '--runtime=nvidia '
    if USE_NVIDIA_RAPIDS:
        DEFAULT_DOCKER_IMAGE = 'activeeon/rapidsai'
    else:
        DEFAULT_DOCKER_IMAGE = 'activeeon/cuda'

# use a different DOCKER_IMAGE is it's set
if variables.get("DOCKER_IMAGE") is not None:
    DOCKER_IMAGE = variables.get("DOCKER_IMAGE")
else:
    DOCKER_IMAGE = DEFAULT_DOCKER_IMAGE

# print the current docker environment
print('Fork environment info...')
print('DOCKER_ENABLED:     ' + str(DOCKER_ENABLED))
print('DOCKER_IMAGE:       ' + DOCKER_IMAGE)
print('CUDA_ENABLED:       ' + str(CUDA_ENABLED))
print('DOCKER_GPU_ENABLED: ' + str(DOCKER_GPU_ENABLED))
print('USE_NVIDIA_RAPIDS:  ' + str(USE_NVIDIA_RAPIDS))
print('DOCKER_RUN_CMD:     ' + DOCKER_RUN_CMD)

if DOCKER_ENABLED == True:
    # Prepare Docker parameters
    containerName = DOCKER_IMAGE
    dockerRunCommand =  DOCKER_RUN_CMD
    dockerParameters = '--rm '
    # Prepare ProActive home volume
    paHomeHost = variables.get("PA_SCHEDULER_HOME")
    paHomeContainer = variables.get("PA_SCHEDULER_HOME")
    proActiveHomeVolume = '-v '+paHomeHost +':'+paHomeContainer+' '
    # Prepare working directory (For Dataspaces and serialized task file)
    workspaceHost = localspace
    workspaceContainer = localspace
    workspaceVolume = '-v '+localspace +':'+localspace+' '
    workspaceShared = ''
    #workspaceShared = '-v /shared:/shared '
    dockerUser = ''
    #dockerUser = '--user 1000:1000 '
    #dockerUser = '--user $(id -u):$(id -g) '
    # Prepare container working directory
    containerWorkingDirectory = '-w '+workspaceContainer+' '
    # Save pre execution command into magic variable 'preJavaHomeCmd', which is picked up by the node
    preJavaHomeCmd = dockerRunCommand + dockerParameters + dockerUser + proActiveHomeVolume + workspaceVolume + workspaceShared + containerWorkingDirectory + containerName

    print('DOCKER_FULL_CMD:    ' + preJavaHomeCmd)
else:
    print("Fork environment disabled")