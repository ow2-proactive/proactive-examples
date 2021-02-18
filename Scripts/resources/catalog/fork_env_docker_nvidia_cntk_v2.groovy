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
if (DOCKER_ENABLED) {
    try {
        Runtime.getRuntime().exec("docker")
    } catch (Exception e) {
        println "Docker does not exists : " + e.getMessage()
        DOCKER_ENABLED = false
    }
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
    cmd.add("--shm-size=256M")

    if (DOCKER_GPU_ENABLED) {
        // Versions earlier than 19.03 require nvidia-docker2 and the --runtime=nvidia flag.
        // On versions including and after 19.03, you will use the nvidia-container-toolkit package
        // and the --gpus all flag.
        try {
            def sout = new StringBuffer(), serr = new StringBuffer()
            def proc = 'docker version -f "{{.Server.Version}}"'.execute()
            proc.consumeProcessOutput(sout, serr)
            proc.waitForOrKill(1000)
            docker_version = sout.toString()
            docker_version = docker_version.substring(1, docker_version.length()-2)
            docker_version_major = docker_version.split("\\.")[0].toInteger()
            docker_version_minor = docker_version.split("\\.")[1].toInteger()
            println "Docker version: " + docker_version
            if ((docker_version_major >= 19) && (docker_version_minor >= 3)) {
                cmd.add("--gpus=all")
            } else {
                cmd.add("--runtime=nvidia")
            }
        } catch (Exception e) {
            println "Error while getting the docker version: " + e.getMessage()
            println "DOCKER_GPU_ENABLED is off"
        }
        // rootless containers leveraging NVIDIA GPUs
        // needed when cgroups is disabled in nvidia-container-runtime
        // /etc/nvidia-container-runtime/config.toml => no-cgroups = true
        cmd.add("--privileged") // https://github.com/NVIDIA/nvidia-docker/issues/1171
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

    paContainerName = System.getProperty("proactive.container.name")
    isPANodeInContainer = (paContainerName != null && !paContainerName.isEmpty())
    paContainerHostAddress = System.getProperty("proactive.container.host.address")

    if (isPANodeInContainer) {
        cmd.add("--volumes-from")
        cmd.add(paContainerName)
        cmd.add("--add-host")
        cmd.add("service-node:" + paContainerHostAddress)
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