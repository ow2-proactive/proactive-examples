// This script creates a docker fork environment using CNTK (CPU + GPU configuration)
// Job or Task variables used:
// DOCKER_ENABLED: True (default)
// MOUNT_LOG_PATH: optional host path to store logs
// DOCKER_LOG_PATH: mounting point of optional logs in the docker container

DOCKER_ENABLED = true
if (variables.get("DOCKER_ENABLED") == null || variables.get("DOCKER_ENABLED").toLowerCase().equals("false")) {
    DOCKER_ENABLED = false
}

if ((new File("/.dockerenv")).exists()) {
    DOCKER_ENABLED = false
}

MOUNT_LOG_PATH = variables.get("MOUNT_LOG_PATH")
DOCKER_LOG_PATH = variables.get("DOCKER_LOG_PATH")

DOCKER_GPU_ENABLED = true
if (variables.get("DOCKER_GPU_ENABLED") != null && variables.get("DOCKER_GPU_ENABLED").toLowerCase().equals("false")) {
    DOCKER_GPU_ENABLED = false
}

CUDA_ENABLED = false
CUDA_HOME = System.getenv('CUDA_HOME')
CUDA_HOME_DEFAULT = "/usr/local/cuda"
if (CUDA_HOME && (new File(CUDA_HOME)).isDirectory()) {
    CUDA_ENABLED = true
} else if ((new File(CUDA_HOME_DEFAULT)).isDirectory()) {
    CUDA_ENABLED = true
}
if (!CUDA_ENABLED) {
    DOCKER_GPU_ENABLED = false
}

DEFAULT_DOCKER_IMAGE = "activeeon/cntk:2.4-cpu-python3.5"
if (DOCKER_GPU_ENABLED) {
    DEFAULT_DOCKER_IMAGE = "activeeon/cntk"
}

if (variables.get("DOCKER_IMAGE") != null && !variables.get("DOCKER_IMAGE").isEmpty()) {
    DOCKER_IMAGE = variables.get("DOCKER_IMAGE")
} else {
    DOCKER_IMAGE = DEFAULT_DOCKER_IMAGE
}

println "Fork environment info..."
println "DOCKER_ENABLED:     " + DOCKER_ENABLED
println "DOCKER_IMAGE:       " + DOCKER_IMAGE
println "DOCKER_GPU_ENABLED: " + DOCKER_GPU_ENABLED
println "CUDA_ENABLED:       " + CUDA_ENABLED

if (DOCKER_ENABLED) {
    // In the Java Home location field, use the value: "/usr" to force using the JRE provided in the docker image below (Recommended).
    // Be aware, that the prefix command is internally split by spaces. So paths with spaces won't work.
    // Prepare Docker parameters
    containerName = DOCKER_IMAGE
    dockerRunCommand = "docker run "
    if (DOCKER_GPU_ENABLED) {
        dockerParameters = "--rm --runtime=nvidia "
    } else {
        dockerParameters = "--rm "
    }
    //dockerParameters = "--rm --env HOME=/tmp "

    // Prepare ProActive home volume
    paHomeHost = variables.get("PA_SCHEDULER_HOME")
    paHomeContainer = variables.get("PA_SCHEDULER_HOME")
    proActiveHomeVolume = "-v " + paHomeHost + ":" + paHomeContainer + " "
    // Prepare working directory (For Dataspaces and serialized task file)
    workspaceHost = localspace
    workspaceContainer = localspace
    workspaceVolume = "-v " + workspaceHost + ":" + workspaceContainer + " "
    
    // Prepare log directory
    logPathVolume = ""
    /*if (MOUNT_LOG_PATH && DOCKER_LOG_PATH) {
        mountLogHost = MOUNT_LOG_PATH
        logPathContainer = DOCKER_LOG_PATH
        logPathVolume = "-v " + mountLogHost + ":" + logPathContainer + " "
    }*/

    // Prepare container working directory
    containerWorkingDirectory = "-w " + workspaceContainer + " "
    
    userDefinition = ""
    /*sigar = new org.hyperic.sigar.Sigar()
    try {
        pid = sigar.getPid()
        creds = sigar.getProcCred(pid)
        uid = creds.getUid()
        gid = creds.getGid()
        userDefinition = "--user=" + uid + ":" + gid + " "
    } catch (Exception e) {
        println "Cannot retrieve user or group id : " + e.getMessage()
    } finally {
        sigar.close()
    }*/

    // Save pre execution command into magic variable 'preJavaHomeCmd', which is picked up by the node
    preJavaHomeCmd = dockerRunCommand + dockerParameters + proActiveHomeVolume + workspaceVolume + logPathVolume + userDefinition + containerWorkingDirectory + containerName

    println "DOCKER_FULL_CMD:    " + preJavaHomeCmd
} else {
    println "Fork environment disabled"
}