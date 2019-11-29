// This script creates a docker fork environment using CNTK (CPU + GPU configuration)
// Job or Task variables used:
// DOCKER_ENABLED: true (default)
// DOCKER_GPU_ENABLED: True to use nvidia gpu libraries (default: false)
// DOCKER_IMAGE: an optional docker image to use

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
if ((new File("/.dockerenv")).exists() && !(new File("/var/run/docker.sock")).exists()) {
    println ("Already inside docker container, without host docker access")
    DOCKER_ENABLED = false
}

DOCKER_GPU_ENABLED = false
if ("true".equals(variables.get("DOCKER_GPU_ENABLED"))) {
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
    containerName = DOCKER_IMAGE
    cmd = []
    cmd.add("docker")
    cmd.add("run")
    cmd.add("--rm")
    if (DOCKER_GPU_ENABLED) {
        cmd.add("--runtime=nvidia")
    }

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

    // Prepare ProActive home volume
    paHomeHost = variables.get("PA_SCHEDULER_HOME")
    paHomeContainer = (isWindows ? forkEnvironment.convertToLinuxPath(paHomeHost) : paHomeHost)
    cmd.add("-v")
    cmd.add(paHomeHost + ":" + paHomeContainer)
    // Prepare working directory (For Dataspaces and serialized task file)
    workspaceHost = localspace
    workspaceContainer = (isWindows ? forkEnvironment.convertToLinuxPath(workspaceHost) : workspaceHost)
    cmd.add("-v")
    cmd.add(workspaceHost + ":" + workspaceContainer)

    cachespaceHost = cachespace
    cachespaceContainer = (isWindows ? forkEnvironment.convertToLinuxPath(cachespaceHost) : cachespaceHost)
    cachespaceHostFile = new File(cachespaceHost)
    if (cachespaceHostFile.exists() && cachespaceHostFile.canRead()) {
        cmd.add("-v")
        cmd.add(cachespaceHost + ":" + cachespaceContainer)
    } else {
        println cachespaceHost + " does not exist or is not readable, access to cache space will be disabled in the container"
    }

    if (!isWindows) {
        // when not on windows, mount and use the current JRE
        currentJavaHome = System.getProperty("java.home")
        forkEnvironment.setJavaHome(currentJavaHome)
        cmd.add("-v")
        cmd.add(currentJavaHome + ":" + currentJavaHome)
    }

    // Prepare container working directory
    cmd.add("-w")
    cmd.add(workspaceContainer)

    cmd.add(containerName)

    forkEnvironment.setPreJavaCommand(cmd)

    // Show the generated command
    println "DOCKER COMMAND : " + forkEnvironment.getPreJavaCommand()
} else {
    println "Fork environment disabled"
}