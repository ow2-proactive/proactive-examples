// This script creates a docker fork environment using activeeon/dlm3 as docker image

// In the Java Home location field, use the value: "/usr" to force using the JRE provided in the docker image below (Recommended).
// Be aware, that the prefix command is internally split by spaces. So paths with spaces won't work.
// Prepare Docker parameters
containerName = "activeeon/dlm3"
if (variables.get("DOCKER_IMAGE") != null && !variables.get("DOCKER_IMAGE").isEmpty()) {
    containerName = variables.get("DOCKER_IMAGE")
}
dockerRunCommand =  "docker run "
dockerParameters = "--rm --env HOME=/tmp "
// Prepare ProActive home volume
paHomeHost = variables.get("PA_SCHEDULER_HOME")
paHomeContainer = variables.get("PA_SCHEDULER_HOME")
proActiveHomeVolume = '-v '+paHomeHost +':'+paHomeContainer+' '
// Prepare working directory (For Dataspaces and serialized task file)
workspaceHost = localspace
workspaceContainer = localspace
workspaceVolume = '-v '+localspace +':'+localspace+' '
// Prepare container working directory
containerWorkingDirectory = '-w '+workspaceContainer+' '

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

// Save pre execution command into magic variable 'preJavaHomeCmd', which is picked up by the node
preJavaHomeCmd = dockerRunCommand + dockerParameters + proActiveHomeVolume + workspaceVolume + userDefinition + containerWorkingDirectory + containerName
println "DOCKER_FULL_CMD:    " + preJavaHomeCmd

