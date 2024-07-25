/*
This script creates a container fork environment for various machine learning usages (CUDA, GPU, RAPIDS ...)
and uses task or job variables for configuration.

Variables:
 - CONTAINER_PLATFORM: docker, podman, singularity or null/empty (none)
 - CONTAINER_IMAGE: container image name (default=activeeon/dlm3)
 - CONTAINER_GPU_ENABLED: true/false, set to false to disable gpu support (default=true)
 - USE_NVIDIA_RAPIDS: true/false, set to true to use activeeon/rapidsai image (default=false)
 - HOST_LOG_PATH: optional host path to store logs
 - CONTAINER_LOG_PATH: mounting point of optional logs in the container
 - CONTAINER_ROOTLESS_ENABLED: true/false, set to false to disable rootless mode (default=true)

If used on windows:
 - currently, only linux containers are supported
 - make sure the drives containing the scheduler installation and TEMP folders are shared with docker containers
 - the container used must have java installed by default in the /usr folder. Change the value of the java home parameter to use a different installation path

On linux, the java installation used by the ProActive Node will be also used inside the container
*/

import org.ow2.proactive.utils.OperatingSystem
import org.ow2.proactive.utils.OperatingSystemFamily
import org.codehaus.groovy.runtime.StackTraceUtils

def CONTAINER_GPU_ENABLED = true
if ("false".equalsIgnoreCase(variables.get("CONTAINER_GPU_ENABLED")) ||
    "false".equalsIgnoreCase(variables.get("DOCKER_GPU_ENABLED"))) {  // backwards compatibility
    CONTAINER_GPU_ENABLED = false
}

def CUDA_ENABLED = false
def CUDA_HOME = System.getenv('CUDA_HOME')
def CUDA_HOME_DEFAULT = "/usr/local/cuda"
if (CUDA_HOME && (new File(CUDA_HOME)).isDirectory()) {
    CUDA_ENABLED = true
} else if ((new File(CUDA_HOME_DEFAULT)).isDirectory()) {
    CUDA_ENABLED = true
}
if (!CUDA_ENABLED) {
    CONTAINER_GPU_ENABLED = false
}

def USE_NVIDIA_RAPIDS = false
if ("true".equalsIgnoreCase(variables.get("USE_NVIDIA_RAPIDS"))) {
    USE_NVIDIA_RAPIDS = true
}

def DEFAULT_CONTAINER_IMAGE = "docker://activeeon/dlm3"

// activate CUDA support if CONTAINER_GPU_ENABLED is True
if (CONTAINER_GPU_ENABLED) {
    if (USE_NVIDIA_RAPIDS) {
        DEFAULT_CONTAINER_IMAGE = "docker://activeeon/rapidsai"
    } else {
        DEFAULT_CONTAINER_IMAGE = "docker://activeeon/cuda"
    }
}

def CONTAINER_IMAGE = DEFAULT_CONTAINER_IMAGE
if (variables.get("CONTAINER_IMAGE") != null && !variables.get("CONTAINER_IMAGE").isEmpty()) {
    CONTAINER_IMAGE = variables.get("CONTAINER_IMAGE")
} else if(variables.get("DOCKER_IMAGE") != null && !variables.get("DOCKER_IMAGE").isEmpty()) {
    CONTAINER_IMAGE = "docker://" + variables.get("DOCKER_IMAGE") // backwards compatibility
} else {
    println "Using the default image: " + CONTAINER_IMAGE
}

def CONTAINER_PLATFORM = variables.get("CONTAINER_PLATFORM")
def SUPPORTED_PLATFORMS = ["docker", "podman", "singularity"]
def CONTAINER_ENABLED = SUPPORTED_PLATFORMS.any{it.equalsIgnoreCase(CONTAINER_PLATFORM)}

if ("true".equalsIgnoreCase(variables.get("DOCKER_ENABLED"))) {
    // backwards compatibility
    CONTAINER_PLATFORM = "docker"
    CONTAINER_ENABLED = true
}
if ((new File("/.dockerenv")).exists() && ! (new File("/var/run/docker.sock")).exists()) {
    println ("Already inside docker container, without host docker access")
    CONTAINER_ENABLED = false
}
if (CONTAINER_ENABLED) {
    try {
        def sout = new StringBuffer(), serr = new StringBuffer()
        def proc = (CONTAINER_PLATFORM + ' --help').execute()
        proc.consumeProcessOutput(sout, serr)
        proc.waitForOrKill(10000)
    } catch (Exception e) {
        println CONTAINER_PLATFORM + " does not exists : " + e.getMessage()
        CONTAINER_ENABLED = false
    }
}

def HOST_LOG_PATH = null
if (variables.get("HOST_LOG_PATH") != null && !variables.get("HOST_LOG_PATH").isEmpty()) {
    HOST_LOG_PATH = variables.get("HOST_LOG_PATH")
} else if (variables.get("MOUNT_LOG_PATH") != null && !variables.get("MOUNT_LOG_PATH").isEmpty()) {
    HOST_LOG_PATH = variables.get("MOUNT_LOG_PATH") // backwards compatibility
}

def CONTAINER_LOG_PATH = null
if (variables.get("CONTAINER_LOG_PATH") != null && !variables.get("CONTAINER_LOG_PATH").isEmpty()) {
    CONTAINER_LOG_PATH = variables.get("CONTAINER_LOG_PATH")
} else if (variables.get("DOCKER_LOG_PATH") != null && !variables.get("DOCKER_LOG_PATH").isEmpty()) {
    CONTAINER_LOG_PATH = variables.get("DOCKER_LOG_PATH") // backwards compatibility
}

CONTAINER_ROOTLESS_ENABLED = true
if ("false".equalsIgnoreCase(variables.get("CONTAINER_ROOTLESS_ENABLED"))) {
    CONTAINER_ROOTLESS_ENABLED = false
}

println "Fork environment info..."
println "CONTAINER_PLATFORM:         " + CONTAINER_PLATFORM
println "CONTAINER_ENABLED:          " + CONTAINER_ENABLED
println "CONTAINER_IMAGE:            " + CONTAINER_IMAGE
println "CONTAINER_GPU_ENABLED:      " + CONTAINER_GPU_ENABLED
println "CUDA_ENABLED:               " + CUDA_ENABLED
println "USE_NVIDIA_RAPIDS:          " + USE_NVIDIA_RAPIDS
println "HOST_LOG_PATH:              " + HOST_LOG_PATH
println "CONTAINER_LOG_PATH:         " + CONTAINER_LOG_PATH
println "CONTAINER_ROOTLESS_ENABLED: " + CONTAINER_ROOTLESS_ENABLED

String osName = System.getProperty("os.name")
println "Operating system : " + osName
OperatingSystem operatingSystem = OperatingSystem.resolveOrError(osName)
OperatingSystemFamily family = operatingSystem.getFamily()

// If docker or podman
if (CONTAINER_ENABLED && (
    "docker".equalsIgnoreCase(CONTAINER_PLATFORM) ||
    "podman".equalsIgnoreCase(CONTAINER_PLATFORM))) {

    if (CONTAINER_IMAGE.startsWith("docker://")) {
        CONTAINER_IMAGE = CONTAINER_IMAGE.substring(CONTAINER_IMAGE.indexOf("://") + 3)
    }

    // Prepare Docker parameters
    containerName = CONTAINER_IMAGE
    cmd = []
    cmd.add(CONTAINER_PLATFORM.toLowerCase())
    cmd.add("run")
    cmd.add("--rm")
    cmd.add("--shm-size=256M")
    // cmd.add("--env")
    // cmd.add("HOME=/tmp")

    if (CUDA_ENABLED && CONTAINER_GPU_ENABLED) {
        if ("docker".equalsIgnoreCase(CONTAINER_PLATFORM)) {
            // Versions earlier than 19.03 require nvidia-docker2 and the --runtime=nvidia flag.
            // On versions including and after 19.03, you will use the nvidia-container-toolkit package
            // and the --gpus all flag.
            try {
                def sout = new StringBuffer(), serr = new StringBuffer()
                def proc = 'docker version -f "{{.Server.Version}}"'.execute()
                proc.consumeProcessOutput(sout, serr)
                proc.waitForOrKill(10000)
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
        } else {
            cmd.add("--runtime=nvidia")
        }
        // rootless containers leveraging NVIDIA GPUs
        // needed when cgroups is disabled in nvidia-container-runtime
        // /etc/nvidia-container-runtime/config.toml => no-cgroups = true
        cmd.add("--privileged") // https://github.com/NVIDIA/nvidia-docker/issues/1171
    }

    isWindows = false
    isMac = false
    switch (family) {
        case OperatingSystemFamily.WINDOWS:
            isWindows = true
            break
        case OperatingSystemFamily.MAC:
            isMac = true
            break
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

    if (!isWindows && !isMac) {
        // when not on windows, mount and use the current JRE
        currentJavaHome = System.getProperty("java.home")
        forkEnvironment.setJavaHome(currentJavaHome)
        if (!isPANodeInContainer) {
            cmd.add("-v")
            cmd.add(currentJavaHome + ":" + currentJavaHome)
        }

        // when not on windows, mount a shared folder if it exists
        sharedDirectory = new File("/shared")
        if (sharedDirectory.isDirectory() && sharedDirectory.canWrite()) {
            if (!isPANodeInContainer) {
                cmd.add("-v")
                cmd.add("/shared:/shared")
            }
        }
    }

    // Prepare log directory
    logPathVolume = ""
    if (HOST_LOG_PATH && CONTAINER_LOG_PATH) {
        mountLogHost = HOST_LOG_PATH
        logPathContainer = CONTAINER_LOG_PATH
        cmd.add("-v")
        cmd.add(mountLogHost + ":" + logPathContainer)
    }

    // Prepare container working directory
    cmd.add("-w")
    cmd.add(workspaceContainer)

    // linux on windows/mac does not allow sharing identities (such as AD identities)
    if (!isWindows && !isMac && CONTAINER_ROOTLESS_ENABLED) {
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
    println "CONTAINER COMMAND : " + forkEnvironment.getPreJavaCommand()
}

// If singularity
if (CONTAINER_ENABLED &&
    "singularity".equalsIgnoreCase(CONTAINER_PLATFORM)) {

    imageUrl = CONTAINER_IMAGE
    imageName = imageUrl.substring(imageUrl.indexOf("://") + 3).replace("/","_").replace(":","_")
    imageLockFileName = imageName + ".lock"
    userHome = System.getProperty("user.home")

    try {
        switch (family) {
            case OperatingSystemFamily.WINDOWS:
            throw new IllegalStateException("Singularity/Apptainer is not supported on Windows operating system")
            case OperatingSystemFamily.MAC:
            throw new IllegalStateException("Singularity/Apptainer is not supported on Mac operating system")
            default:
                isWindows = false;
        }

        process = "singularity --version".execute()
        majorVersion = 0
        if (process.waitFor() == 0) {
            version = process.text
            version = version.replace("singularity version ", "").trim()
            version = version.replace("apptainer version ", "").trim()
            println "Singularity version: " + version
            majorVersion = Integer.parseInt(version.substring(0, version.indexOf(".")))
        } else {
            throw new IllegalStateException("Cannot find singularity command")
        }

        if (majorVersion >= 3) {
            imageFile = imageName + ".sif"
        } else {
            imageFile = imageName + ".simg"
        }

        // synchronization to avoid concurrent NFS conflicts when creating the image
        imageLockPath = new File(userHome, imageLockFileName);
        if (!imageLockPath.exists()) {
            imageLockPath.createNewFile()
        } else {
            while (imageLockPath.exists()) {
                Thread.sleep(1000)
            }
        }

        // pull the container inside the synchronization lock
        if (majorVersion >= 3) {
            pullCmd = "singularity pull --dir " + userHome + " " + imageFile + " " + imageUrl
            println pullCmd
        	def env = System.getenv().collect { k, v -> "$k=$v" }
        	env.push('XDG_RUNTIME_DIR=/run/user/$UID')
        	process = pullCmd.execute(env, new File(userHome))
        } else {
            pullCmd = "singularity pull --name " + imageFile + " " + imageUrl
            def env = System.getenv().collect { k, v -> "$k=$v" }
            env.push("SINGULARITY_PULLFOLDER=" + userHome)
            process = pullCmd.execute(env, new File(userHome))
        }

        process.in.eachLine { line ->
            println line
        }
        process.waitFor()

        (new File(userHome, imageFile)).setReadable(true, false)

        // release lock
        imageLockPath.delete()

        cmd = []
        cmd.add("singularity")
        cmd.add("exec")

        if (CUDA_ENABLED && CONTAINER_GPU_ENABLED) {
            cmd.add("--nv") // add gpu support
        }

        forkEnvironment.setDockerWindowsToLinux(isWindows)

        // Prepare ProActive home volume
        paHomeHost = variables.get("PA_SCHEDULER_HOME")
        paHomeContainer = (isWindows ? forkEnvironment.convertToLinuxPath(paHomeHost) : paHomeHost)
        cmd.add("-B")
        cmd.add(paHomeHost + ":" + paHomeContainer)
        // Prepare working directory (For Dataspaces and serialized task file)
        workspaceHost = localspace
        workspaceContainer = (isWindows ? forkEnvironment.convertToLinuxPath(workspaceHost) : workspaceHost)
        cmd.add("-B")
        cmd.add(workspaceHost + ":" + workspaceContainer)

        cachespaceHost = cachespace
        cachespaceContainer = (isWindows ? forkEnvironment.convertToLinuxPath(cachespaceHost) : cachespaceHost)
        cachespaceHostFile = new File(cachespaceHost)
        if (cachespaceHostFile.exists() && cachespaceHostFile.canRead()) {
            cmd.add("-B")
            cmd.add(cachespaceHost + ":" + cachespaceContainer)
        } else {
            println cachespaceHost + " does not exist or is not readable, access to cache space will be disabled in the container"
        }

        if (!isWindows) {
            // when not on windows, mount and use the current JRE
            currentJavaHome = System.getProperty("java.home")
            forkEnvironment.setJavaHome(currentJavaHome)
            cmd.add("-B")
            cmd.add(currentJavaHome + ":" + currentJavaHome)
        }

        // Prepare container working directory
        cmd.add("--pwd")
        cmd.add(workspaceContainer)
        cmd.add((new File(userHome, imageFile)).getAbsolutePath())
        
        forkEnvironment.setPreJavaCommand(cmd)
        forkEnvironment.addSystemEnvironmentVariable("XDG_RUNTIME_DIR",'/run/user/$UID')

        // Show the generated command
        println "SINGULARITY COMMAND : " + forkEnvironment.getPreJavaCommand()
    } catch (Exception e) {
        StackTraceUtils.deepSanitize(e)
        e.stackTrace.head().lineNumber
        throw e
    } finally {
        (new File(userHome, imageLockFileName)).delete()
    }
}

if (!CONTAINER_ENABLED) {
    println "Fork environment disabled"
}
