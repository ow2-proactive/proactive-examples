// This script creates a docker fork environment using CNTK (CPU + GPU configuration)
// Job or Task variables used:
// DOCKER_ENABLED: True (default)
// MOUNT_LOG_PATH: optional host path to store logs
// DOCKER_LOG_PATH: mounting point of optional logs in the docker container

// If used on windows:
//  - currently, only linux containers are supported
//  - make sure the drives containing the scheduler installation and TEMP folders are shared with docker containers
//  - the container used must have java installed by default in the /usr folder. Change the value of the java home parameter to use a different installation path
// On linux, the java installation used by the ProActive Node will be also used inside the container

import org.ow2.proactive.utils.OperatingSystem;
import org.ow2.proactive.utils.OperatingSystemFamily;

DOCKER_ENABLED = false
if ("true".equalsIgnoreCase(variables.get("DOCKER_ENABLED"))) {
    DOCKER_ENABLED = true
}
if ((new File("/.dockerenv")).exists() && ! (new File("/var/run/docker.sock")).exists()) {
    println ("Already inside docker container, without host docker access")
    DOCKER_ENABLED = false
}
if (DOCKER_ENABLED) {
    try {
        Runtime.getRuntime().exec("docker")
    } catch (Exception e) {
        println "Docker does not exists : " + e.getMessage()
        DOCKER_ENABLED = false
    }
}

DOCKER_GPU_ENABLED = false
DOCKER_NVIDIA_PATH = "/usr/bin/nvidia-docker"
nVidiaExecutable = new File(DOCKER_NVIDIA_PATH)
if (nVidiaExecutable.exists() && nVidiaExecutable.canExecute()) {
    DOCKER_GPU_ENABLED = true
}

CUDA_ENABLED = false
CUDA_HOME = System.getenv('CUDA_HOME')
CUDA_HOME_DEFAULT = "/usr/local/cuda"
if (CUDA_HOME && (new File(CUDA_HOME)).isDirectory()) {
    CUDA_ENABLED = true
} else if ((new File(CUDA_HOME_DEFAULT)).isDirectory()) {
    CUDA_ENABLED = true
}

// DOCKER_IMAGE = "microsoft/cntk:2.4-cpu-python3.5" # default
DOCKER_IMAGE = "activeeon/cntk:2.4-cpu-python3.5"  // added oracle-java 1.8.0_161 + py4j

if (DOCKER_GPU_ENABLED && CUDA_ENABLED) {
    // DOCKER_IMAGE = "microsoft/cntk" # default
    DOCKER_IMAGE = "activeeon/cntk"  // added oracle-java 1.8.0_161 + py4j
}

MOUNT_LOG_PATH = variables.get("MOUNT_LOG_PATH")
DOCKER_LOG_PATH = variables.get("DOCKER_LOG_PATH")

println "Fork environment info..."
println "DOCKER_ENABLED:     " + DOCKER_ENABLED
println "DOCKER_IMAGE:       " + DOCKER_IMAGE
println "DOCKER_GPU_ENABLED: " + DOCKER_GPU_ENABLED
println "CUDA_ENABLED:       " + CUDA_ENABLED

if (DOCKER_ENABLED) {
    // Prepare Docker parameters
    containerName = DOCKER_IMAGE
    cmd = []

    if (DOCKER_GPU_ENABLED && CUDA_ENABLED) {
        cmd.add("nvidia-docker")
    } else {
        cmd.add("docker")
    }
    cmd.add("run")
    cmd.add("--rm")
    cmd.add("--shm-size=256M")
    cmd.add("--env")
    cmd.add("HOME=/tmp")

    String osName = System.getProperty("os.name");
    println "Operating system : " + osName;
    OperatingSystem operatingSystem = OperatingSystem.resolveOrError(osName);
    OperatingSystemFamily family = operatingSystem.getFamily();

    switch (family) {
        case OperatingSystemFamily.WINDOWS:
            isWindows = true;
            break;
        default:
            isWindows = false;
    }
    forkEnvironment.setDockerWindowsToLinux(isWindows)

    paContainerName = System.getProperty("proactive.container.name")
    isPANodeInContainer = (paContainerName != null && !paContainerName.isEmpty())

    if (isPANodeInContainer) {
        cmd.add("--volumes-from")
        cmd.add(paContainerName)
    }

    // Prepare ProActive home volume
    paHomeHost = variables.get("PA_SCHEDULER_HOME")
    paHomeContainer = (isWindows ? forkEnvironment.convertToLinuxPath(paHomeHost) : paHomeHost)
    if (!isPANodeInContainer) {
        cmd.add("-v")
        cmd.add(paHomeHost + ":" + paHomeContainer)
    }
    // Prepare working directory (For Dataspaces and serialized task file)
    workspaceHost = localspace
    workspaceContainer = (isWindows ? forkEnvironment.convertToLinuxPath(workspaceHost) : workspaceHost)
    if (!isPANodeInContainer) {
        cmd.add("-v")
        cmd.add(workspaceHost + ":" + workspaceContainer)
    }

    cachespaceHost = cachespace
    cachespaceContainer = (isWindows ? forkEnvironment.convertToLinuxPath(cachespaceHost) : cachespaceHost)
    cachespaceHostFile = new File(cachespaceHost)
    if (cachespaceHostFile.exists() && cachespaceHostFile.canRead()) {
        if (!isPANodeInContainer) {
            cmd.add("-v")
            cmd.add(cachespaceHost + ":" + cachespaceContainer)
        }
    } else {
        println cachespaceHost + " does not exist or is not readable, access to cache space will be disabled in the container"
    }

    if (!isWindows) {
        // when not on windows, mount and use the current JRE
        currentJavaHome = System.getProperty("java.home")
        forkEnvironment.setJavaHome(currentJavaHome)
        if (!isPANodeInContainer) {
            cmd.add("-v")
            cmd.add(currentJavaHome + ":" + currentJavaHome)
        }
    }

    // Prepare log directory
    logPathVolume = ""
    if (MOUNT_LOG_PATH && DOCKER_LOG_PATH) {
        mountLogHost = MOUNT_LOG_PATH
        logPathContainer = DOCKER_LOG_PATH
        if (!isPANodeInContainer) {
            cmd.add("-v")
            cmd.add(mountLogHost + ":" + logPathContainer)
        }
    }

    // Prepare container working directory
    cmd.add("-w")
    cmd.add(workspaceContainer)

    sigar = new org.hyperic.sigar.Sigar()
    try {
        pid = sigar.getPid()
        creds = sigar.getProcCred(pid)
        uid = creds.getUid()
        gid = creds.getGid()
        userDefinition = "--user=" + uid + ":" + gid + " "
    } catch (Exception e) {
        println "Cannot retrieve user or group id : " + e.getMessage()
        userDefinition = "";
    } finally {
        sigar.close()
    }

    cmd.add(containerName)

    forkEnvironment.setPreJavaCommand(cmd)

    // Show the generated command
    println "DOCKER COMMAND : " + forkEnvironment.getPreJavaCommand()
} else {
    println "Fork environment disabled"
}
