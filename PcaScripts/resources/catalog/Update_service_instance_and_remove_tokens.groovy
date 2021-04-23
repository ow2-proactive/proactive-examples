import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import com.google.common.base.Strings;

// 2 ways to pass script arguments
if (new File(localspace, "arguments.txt").exists()) {
    arguments_array = new File(localspace, "arguments.txt").text.split(",")
} else {
    arguments_array = args
}

if (arguments_array.length != 2) {
    println("[Update_service_instance_and_remove_tokens] ERROR Number of arguments must be == 2")
    System.exit(1)
}

def i = 0
def service_instance_status = arguments_array[i++].trim()
def token_name = arguments_array[i++].trim()

println "[Update_service_instance_and_remove_tokens] service_instance_status " + service_instance_status
println "[Update_service_instance_and_remove_tokens] token_name " + token_name

def is_defined(param) {
    !Strings.isNullOrEmpty(param) && !param.toLowerCase().equals("null")
}

// Retrieve variables
def instance_id = variables.get("PCA_INSTANCE_ID") as long
def pca_url = variables.get('PA_CLOUD_AUTOMATION_REST_URL')

// Get schedulerapi access and acquire session id
schedulerapi.connect()
def sessionId = schedulerapi.getSession()

// Connect to Cloud Automation API
def service_instance_rest_api = new ServiceInstanceRestApi(new ApiClient().setBasePath(pca_url))

// Get service instance
def service_instance_data = service_instance_rest_api.getServiceInstanceUsingGET(sessionId, instance_id)

// Set the service instance status
if (is_defined(service_instance_status)){
    service_instance_data.setInstanceStatus(service_instance_status)
}

// Update service instance
service_instance_data = service_instance_rest_api.updateServiceInstanceUsingPUT(sessionId, instance_id, service_instance_data)

// Remove all node tokens
if (is_defined(token_name)) {

    // Remove tokens if the main job is finished
    def submitted_main_job_id = service_instance_data.getJobSubmissions().get(0).getJobId()
    def is_finished = schedulerapi.getJobState(submitted_main_job_id.toString()).isFinished()
    println "[Update_service_instance_and_remove_tokens] job " + submitted_main_job_id + " is_finished? " + is_finished
    if (is_finished) {

        rmapi.connect()
        def deploymentsIterator = service_instance_data.getDeployments().iterator()
        i = 0
        while (deploymentsIterator.hasNext()) {
            def pa_node_url_to_remove_token = deploymentsIterator.next().getNode().getUrl()
            println "[Update_service_instance_and_remove_tokens] Removing token " + token_name + " from node " + pa_node_url_to_remove_token
            rmapi.removeNodeToken(pa_node_url_to_remove_token, token_name)
            i++
        }
    }
}