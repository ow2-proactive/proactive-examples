// This script creates a docker fork environment for various machine learning usages (CUDA, GPU, RAPIDS ...) and uses task or job variables for configuration.
// Variables:
// DOCKER_ENABLED: true/false, set to false to disable docker completely (default=true)
// DOCKER_IMAGE: docker image name (default=activeeon/dlm3)
// DOCKER_GPU_ENABLED: true/false, set to true to enable gpu parameters and use activeeon/cuda image (default=false)
// USE_NVIDIA_RAPIDS: true/false, set to true to use activeeon/rapidsai image (default=false)
// MOUNT_LOG_PATH: optional host path to store logs
// DOCKER_LOG_PATH: mounting point of optional logs in the docker container

// If used on windows:
//  - currently, only linux containers are supported
//  - make sure the drives containing the scheduler installation and TEMP folders are shared with docker containers
//  - the container used must have java installed by default in the /usr folder. Change the value of the java home parameter to use a different installation path
// On linux, the java installation used by the ProActive Node will be also used inside the container

import org.ow2.proactive.utils.OperatingSystem
import org.ow2.proactive.utils.OperatingSystemFamily

DOCKER_ENABLED = true
if ("false".equalsIgnoreCase(variables.get("DOCKER_ENABLED"))) {
    DOCKER_ENABLED = false
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

MOUNT_LOG_PATH = variables.get("MOUNT_LOG_PATH")
DOCKER_LOG_PATH = variables.get("DOCKER_LOG_PATH")

DOCKER_GPU_ENABLED = false
if (variables.get("DOCKER_GPU_ENABLED") != null && variables.get("DOCKER_GPU_ENABLED").toLowerCase().equals("true")) {
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

USE_NVIDIA_RAPIDS = false
if (variables.get("USE_NVIDIA_RAPIDS") != null && variables.get("USE_NVIDIA_RAPIDS").toLowerCase().equals("true")) {
    USE_NVIDIA_RAPIDS = true
}

DEFAULT_DOCKER_IMAGE = "activeeon/dlm3"

// activate CUDA support if DOCKER_GPU_ENABLED is True
if (DOCKER_GPU_ENABLED) {
    if (USE_NVIDIA_RAPIDS) {
        DEFAULT_DOCKER_IMAGE = "activeeon/rapidsai"
    } else {
        DEFAULT_DOCKER_IMAGE = "activeeon/cuda"
    }
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
println "USE_NVIDIA_RAPIDS: "  + USE_NVIDIA_RAPIDS

if (DOCKER_ENABLED) {
    // Prepare Docker parameters
    containerName = DOCKER_IMAGE
    cmd = []
    cmd.add("docker")
    cmd.add("run")
    cmd.add("--rm")
    cmd.add("--env")
    cmd.add("HOME=/tmp")

    if (CUDA_ENABLED && DOCKER_GPU_ENABLED) {
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

    String osName = System.getProperty("os.name")
    println "Operating system : " + osName
    OperatingSystem operatingSystem = OperatingSystem.resolveOrError(osName)
    OperatingSystemFamily family = operatingSystem.getFamily()

    switch (family) {
        case OperatingSystemFamily.WINDOWS:
            isWindows = true
            break
        default:
            isWindows = false
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

    // Prepare log directory
    logPathVolume = ""
    if (MOUNT_LOG_PATH && DOCKER_LOG_PATH) {
        mountLogHost = MOUNT_LOG_PATH
        logPathContainer = DOCKER_LOG_PATH
        cmd.add("-v")
        cmd.add(mountLogHost + ":" + logPathContainer)
    }

    sharedDirectory = new File("/shared")
    if (sharedDirectory.isDirectory() && sharedDirectory.canWrite()) {
        cmd.add("-v")
        cmd.add("/shared:/shared")
    }

    // Prepare container working directory
    cmd.add("-w")
    cmd.add(workspaceContainer)

    if (isWindows) {
        // linux on windows does not allow sharing identities (such as AD identities)
    } else {
        sigar = new org.hyperic.sigar.Sigar()
        try {
            pid = sigar.getPid()
            creds = sigar.getProcCred(pid)
            uid = creds.getUid()
            gid = creds.getGid()
            cmd.add("--user=" + uid + ":" + gid)
        } catch (Exception e) {
            println "Cannot retrieve user or group id : " + e.getMessage()
        } finally {
            sigar.close()
        }
    }

    cmd.add(containerName)

    forkEnvironment.setPreJavaCommand(cmd)

    // Show the generated command
    println "DOCKER COMMAND : " + forkEnvironment.getPreJavaCommand()
} else {
    println "Fork environment disabled"
}
