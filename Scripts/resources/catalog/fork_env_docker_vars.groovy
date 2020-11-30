// This script creates a docker fork environment configured using job or task variables
// Variables:
// DOCKER_ENABLED: true/false, set to false to disable docker completely (default=false)
// DOCKER_IMAGE: docker image name (default=java)

// If used on windows:
//  - currently, only linux containers are supported
//  - make sure the drives containing the scheduler installation and TEMP folders are shared with docker containers
//  - the container used must have java installed by default in the /usr folder. Change the value of the java home parameter to use a different installation path
// On linux, the java installation used by the ProActive Node will be also used inside the container
import org.ow2.proactive.utils.OperatingSystem
import org.ow2.proactive.utils.OperatingSystemFamily

DOCKER_ENABLED = "true".equalsIgnoreCase(variables.get("DOCKER_ENABLED"))
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

if (DOCKER_ENABLED) {
    // Prepare Docker parameters
    containerName = "java"
    if (variables.get("DOCKER_IMAGE") != null && !variables.get("DOCKER_IMAGE").isEmpty()) {
        containerName = variables.get("DOCKER_IMAGE")
    }

    cmd = []
    cmd.add("docker")
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
