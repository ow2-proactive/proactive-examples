import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.pca.service.client.model.ServiceInstanceData

def pcaUrl = variables.get('PA_CLOUD_AUTOMATION_REST_URL')
def instanceId = variables.get("PCA_INSTANCE_ID") as long
def instanceName = variables.get("INSTANCE_NAME")
def channel = "Service_Instance_" + instanceId

// Get schedulerapi access and acquire session id
schedulerapi.connect()
def sessionId = schedulerapi.getSession()

// Connect to Cloud Automation API
def serviceInstanceRestApi = new ServiceInstanceRestApi(new ApiClient().setBasePath(pcaUrl))

// If service instance is FINISHED or PAUSED then stop this loop and job and delete the sync channel
def currentStatus = serviceInstanceRestApi.getServiceInstanceUsingGET(sessionId, instanceId).getInstanceStatus()
if (currentStatus.equals("FINISHED")){
    variables.put("IS_FINISHED",true)
    synchronizationapi.deleteChannel(channel)
    // Remove token in the current node
    token = instanceName
    nodeUrl = variables.get("PA_NODE_URL")
    println("Current nodeUrl: " + nodeUrl)
    println("Removing token:  " + token)
    rmapi.connect()
    rmapi.removeNodeToken(nodeUrl, token)
} else {
    // Check if container has been stopped abnormally
    def command = ["singularity", "instance", "list"].execute() | ["grep", "${instanceName}"].execute()
    command.waitFor()
    def commandOutput = command.getText()
    def isContainerRunning = commandOutput != ""
    if (!isContainerRunning && (!synchronizationapi.get(channel, "FINISH_LAUNCHED")) && (!synchronizationapi.get(channel, "PAUSE_LAUNCHED"))){
        currentStatus = 'ERROR'
        println("[ERROR] An internal error occured in singularity container: " + instanceName)
        // Update docker container is not running
        def serviceInstanceData = serviceInstanceRestApi.getServiceInstanceUsingGET(sessionId, instanceId)
        serviceInstanceData.setInstanceStatus(currentStatus)
        serviceInstanceRestApi.updateServiceInstanceUsingPUT(sessionId, instanceId, serviceInstanceData)
        // Tell the CRON loop to stop
        variables.put("IS_FINISHED",true)
        // Exit with error
        System.exit(1)
    }
}