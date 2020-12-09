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

def display = args[0]
def visu_command = args[1]

def processVisu = visu_command.execute()
processVisu.consumeProcessOutput(System.out, System.err)
Thread.sleep(1000)
grepProc = 'ps -aux'.execute() | ['grep', visu_command].execute() | 'grep -v grep'.execute() | ['awk', '{ print $2 }'].execute()
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