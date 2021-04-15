import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import com.google.common.base.Strings;

// 2 ways to pass script arguments
if (new File(localspace, "arguments.txt").exists()) {
    arguments_array = new File(localspace, "arguments.txt").text.split(",")
} else {
    arguments_array = args
}

def i = 0
service_instance_status = arguments_array[i++].trim()
token_name = arguments_array[i++].trim()

println "From Update_service_instance_and_remove_token"
println "service_instance_status " + service_instance_status
println "token_name " + token_name

// Retrieve variables
def instance_id = variables.get("PCA_INSTANCE_ID") as long
def pca_url = variables.get('PA_CLOUD_AUTOMATION_REST_URL')

// Get schedulerapi access and acquire session id
schedulerapi.connect()
def sessionId = schedulerapi.getSession()

// Connect to Cloud Automation API
def serviceInstanceRestApi = new ServiceInstanceRestApi(new ApiClient().setBasePath(pca_url))

// Get service instance
def serviceInstanceData = serviceInstanceRestApi.getServiceInstanceUsingGET(sessionId, instance_id)

// Set the service instance status
if (!Strings.isNullOrEmpty(service_instance_status)){
    serviceInstanceData.setInstanceStatus(service_instance_status)
}

// Update service instance
serviceInstanceData = serviceInstanceRestApi.updateServiceInstanceUsingPUT(sessionId, instance_id, serviceInstanceData)
println(serviceInstanceData)

// Remove all node tokens
if (!Strings.isNullOrEmpty(token_name)) {
    rmapi.connect()
    def deploymentsIterator = serviceInstanceData.getDeployments().iterator()
    while (deploymentsIterator.hasNext()) {
        def pa_node_url_to_remove_token = deploymentsIterator.next().getNode().getUrl()
        println "Removing token " + token_name + " from node " + pa_node_url_to_remove_token
        rmapi.removeNodeToken(pa_node_url_to_remove_token, token_name)
    }
}