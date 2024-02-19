import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi

// 2 ways to pass script arguments
if (new File(localspace, "arguments.txt").exists()) {
    arguments_array = new File(localspace, "arguments.txt").text.split(",")
} else {
    arguments_array = args
}

if (arguments_array.length != 4) {
    println("[Loop_over_service_instance_status] ERROR Number of arguments must be == 4")
    throw new IllegalArgumentException("Number of arguments must be 4")
}

def i = 0
def status_is_ok = Boolean.parseBoolean(arguments_array[i++])
def is_docker_based_service = Boolean.parseBoolean(arguments_array[i++])
def token_name = arguments_array[i++]
def container_name = arguments_array[i++]


// Retrieve variables
def pca_url = variables.get('PA_CLOUD_AUTOMATION_REST_URL')
def instance_id = variables.get("PCA_INSTANCE_ID") as long
def channel = "Service_Instance_" + instance_id
def credentials_key = variables.get("CREDENTIALS_KEY")

// Get schedulerapi access and acquire session id
schedulerapi.connect()
def session_id = schedulerapi.getSession()

// Connect to Cloud Automation API
def service_instance_rest_api = new ServiceInstanceRestApi(new ApiClient().setBasePath(pca_url))

// If service instance is FINISHED or PAUSED then stop this loop and job and delete the sync channel
def service_instance_data = service_instance_rest_api.getServiceInstance(session_id, instance_id)
def current_status = service_instance_data.getInstanceStatus()

if (current_status.equals("FINISHED")){

    println "[Loop_over_service_instance_status] IF current_status FINISHED"

    // Break the CRON loop
    variables.put("IS_FINISHED",true)

    // Remove credentials
    if(credentials_key){
        schedulerapi.removeThirdPartyCredential(credentials_key)
    }

    // Delete the sync channel
    synchronizationapi.deleteChannel(channel)

    // detach service to the current and parent job
    schedulerapi.detachService(variables.get("PA_JOB_ID"), instance_id as int)
    if (genericInformation.containsKey("PARENT_JOB_ID") && !schedulerapi.isJobFinished(genericInformation.get("PARENT_JOB_ID"))) {
        try {
            schedulerapi.detachService(genericInformation.get("PARENT_JOB_ID"), instance_id as int)
        } catch (Exception e) {
            // for the rare case where parent job just terminated
            println "[Loop_over_service_instance_status] WARN: could not detach service from job " + genericInformation.get("PARENT_JOB_ID") + " : " + e.getMessage()
        }
    }

    // Remove all tokens
    rmapi.connect()
    def deploymentsIterator = service_instance_data.getDeployments().iterator()
    while (deploymentsIterator.hasNext()) {
        def pa_node_url_to_remove_token = deploymentsIterator.next().getNode().getUrl()
        println "[Loop_over_service_instance_status] Removing token " + token_name + " from node " + pa_node_url_to_remove_token
        rmapi.removeNodeToken(pa_node_url_to_remove_token, token_name)
    }

} else {

    println "[Loop_over_service_instance_status] ELSE if status_is_ok " + status_is_ok + " sync.get(FINISHED_LAUNCHED) " + synchronizationapi.get(channel, "FINISH_LAUNCHED") + " sync.get(PAUSE_LAUNCHED) " + synchronizationapi.get(channel, "PAUSE_LAUNCHED")
    if (!status_is_ok && (!synchronizationapi.get(channel, "FINISH_LAUNCHED")) && (!synchronizationapi.get(channel, "PAUSE_LAUNCHED"))){
        current_status = 'ERROR'

        // Update service instance status
        def serviceInstanceData = service_instance_rest_api.getServiceInstance(session_id, instance_id)
        serviceInstanceData.setInstanceStatus(current_status)
        service_instance_rest_api.updateServiceInstance(session_id, instance_id, serviceInstanceData)

        // Break the CRON loop
        variables.put("IS_FINISHED",true)

        // Exit with error
        throw new IllegalStateException("Service status is in error")

    } else if (is_docker_based_service) {
        // Fetch all logs or only new logs since last fetch time mark
        def lastTime=variables.get('LAST_TIME_MARKER')
        def fetchLogsCmd = lastTime ? ["docker", "logs", "--since", lastTime, container_name] : ["docker", "logs", container_name]
        fetchLogsCmd.execute().waitForProcessOutput(System.out, System.err)
    }
}