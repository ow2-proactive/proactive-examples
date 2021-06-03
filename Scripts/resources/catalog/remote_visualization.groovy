/**
 * Pre-Script which enables Remote Visualization using a VNC command and X terminal session
 *
 * This script creates the variable 'VISU_PID' containing the process id of the VNC session started
 * This session should be terminated with a kill command (otherwise the task will remain running)
 *
 * Arguments:
 * display : X display number (e.g. 12)
 * visu_command : command used to start the VNC session (e.g. "Xvnc :${DISPLAY} -geometry 1280x1024 -SecurityTypes None")
 */

// Generate random port for remote visualization
def findPort() {
    findPortScript = new File("find_port.sh")
    findPortScript << '''
    while :
        do
            RND_PORT="`shuf -i 5911-5999 -n 1`"
	        ss -lpn | grep -q ":$RND_PORT " || break
        done
    echo $RND_PORT
    '''
    'chmod u+x find_port.sh'.execute().text
    port = "./find_port.sh".execute().text
    findPortScript.delete() 
    return port
}

// Prepare remote visualization command
def visu_command = args[0]
port = findPort()
display = port.trim().substring(2)
variables.put("DISPLAY", display)
visu_command_final = visu_command.replace('$DISPLAY',display).replace('${DISPLAY}',display)
println "visu_command = " + visu_command_final

// Start remote visualization
def processVisu = visu_command_final.execute()
processVisu.consumeProcessOutput(System.out, System.err)
Thread.sleep(1000)
grepProc = 'ps -aux'.execute() | ['grep', visu_command_final].execute() | 'grep -v grep'.execute() | ['awk', '{ print $2 }'].execute()
grepProc.waitFor()
visu_pidText = grepProc.text

println "Visu process id: " + visu_pidText

try {
    Integer visuPid = visu_pidText.trim() as Integer
    variables.put("VISU_PID", visuPid)
} catch (Exception e) {
    throw new IllegalStateException("Visu process cannot be found", e)
}    

remoteConnectionString = String.format("PA_REMOTE_CONNECTION;%s;%s;vnc;%s:59%s", variables.get("PA_JOB_ID"), variables.get("PA_TASK_ID"), variables.get("PA_NODE_HOST"), display)
println remoteConnectionString
    
schedulerapi.connect()
schedulerapi.enableRemoteVisualization(variables.get("PA_JOB_ID"), variables.get("PA_TASK_NAME"), remoteConnectionString)

