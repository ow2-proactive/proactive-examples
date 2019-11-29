import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.pca.service.client.model.ServiceInstanceData

def paSchedulerRestUrl = variables.get('PA_SCHEDULER_REST_URL')
def pcaUrl = paSchedulerRestUrl.replaceAll("/rest\\z", "/cloud-automation-service")
def instanceId = variables.get("PCA_INSTANCE_ID") as long
def instanceName = variables.get("INSTANCE_NAME")
def channel = "Service_Instance_" + instanceId

// Connect to Cloud Automation API
def serviceInstanceRestApi = new ServiceInstanceRestApi(new ApiClient().setBasePath(pcaUrl))
def pcaStatesGI = genericInformation.get("pca.states")

// If service instance is FINISHED or PAUSED then stop this loop and job and delete the sync channel
def currentStatus = serviceInstanceRestApi.getServiceInstanceUsingGET(instanceId).getInstanceStatus()
if (currentStatus.equals("FINISHED") || currentStatus.equals("PAUSED") || (pcaStatesGI.equals("(VOID,RUNNING)") && synchronizationapi.get(channel, "RESUMED"))){
    variables.put("IS_FINISHED",true)
    if (currentStatus.equals("FINISHED")){
        synchronizationapi.deleteChannel(channel)
        // Remove token in the current node
        token = instanceName
        nodeUrl = variables.get("PA_NODE_URL")
        println("Current nodeUrl: " + nodeUrl)
        println("Removing token:  " + token)
        rmapi.connect()
        rmapi.removeNodeToken(nodeUrl, token)
    }
} else {
    // Check if container has been stopped abnormally
    def isContainerRunning = ["docker", "inspect", "--format", "{{ .State.Running }}", "${instanceName}"].execute().getText().trim().toBoolean()
    if ((!isContainerRunning) && (!synchronizationapi.get(channel, "FINISH_LAUNCHED")) && (!synchronizationapi.get(channel, "PAUSE_LAUNCHED"))){
        currentStatus = 'ERROR'
        println("[ERROR] An internal error occured in docker container: " + instanceName)
        // Update docker container is not running
        def serviceInstanceData = serviceInstanceRestApi.getServiceInstanceUsingGET(instanceId)
        serviceInstanceData.setInstanceStatus(currentStatus)
        serviceInstanceRestApi.updateServiceInstanceUsingPUT(instanceId, serviceInstanceData)
        // Tell the CRON loop to stop
        variables.put("IS_FINISHED",true)
        // Exit with error
        System.exit(1)
    } else {
        // Fetch all logs or only new logs since last fetch time mark
        def lastTime=variables.get('LAST_TIME_MARKER')
        def fetchLogsCmd = lastTime ? ["docker", "logs", "--since", lastTime, instanceName] : ["docker", "logs", instanceName]
        fetchLogsCmd.execute().waitForProcessOutput(System.out, System.err)
    }
}