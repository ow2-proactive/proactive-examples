// This script creates a docker fork environment using r-base and a data directory

// If used on windows:
//  - currently, only linux containers are supported
//  - make sure the drives containing the scheduler installation and TEMP folders are shared with docker containers
//  - the container used must have java installed by default in the /usr folder. Change the value of the java home parameter to use a different installation path
// On linux, the java installation used by the ProActive Node will be also used inside the container

// Prepare Docker parameters
import org.ow2.proactive.utils.OperatingSystem;
import org.ow2.proactive.utils.OperatingSystemFamily;

containerName = "activeeon/r-base:latest"
cmd = []
cmd.add("docker")
cmd.add("run")
cmd.add("--rm")
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
cmd.add("-v")
cmd.add(cachespaceHost + ":" + cachespaceContainer)

dataDirHost = variables.get("data_dir_path")
dataDirContainer = "/tmp/data"
cmd.add("-v")
cmd.add(dataDirHost + ":" + dataDirContainer)

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

