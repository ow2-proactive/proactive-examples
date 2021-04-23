import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.pca.service.client.model.Deployment
import org.ow2.proactive.pca.service.client.model.Container
import org.ow2.proactive.pca.service.client.model.Endpoint
import org.ow2.proactive.pca.service.client.model.Node
import org.codehaus.groovy.runtime.StackTraceUtils
import com.google.common.base.Strings;

// 2 ways to pass script arguments
if (new File(localspace, "arguments.txt").exists()) {
    arguments_array = new File(localspace, "arguments.txt").text.split(",")
} else {
    arguments_array = args
}

if (arguments_array.length != 9) {
    println("[ERROR] Number of arguments must be == 9")
    System.exit(1)
}

def i = 0
def endpoint_id = arguments_array[i++].trim()
def endpoint_url = arguments_array[i++].trim()
def container_id = arguments_array[i++].trim()
def container_name = arguments_array[i++].trim()
def proxyfied_url = arguments_array[i++].trim()
def service_instance_status = arguments_array[i++].trim()
def register_service = Boolean.parseBoolean(arguments_array[i++])
def inform_service_running = Boolean.parseBoolean(arguments_array[i++])
def token_name = arguments_array[i++].trim()

println "From Update_service_instance_and_add_token"
println "endpoint_id " + endpoint_id
println "endpoint_url " + endpoint_url
println "container_id " + container_id
println "container_name " + container_name
println "proxyfied_url " + proxyfied_url
println "service_instance_status " + service_instance_status
println "register_service " + register_service
println "inform_service_running " + inform_service_running
println "token_name " + token_name

def is_defined(param) {
    !Strings.isNullOrEmpty(param) && !param.toLowerCase().equals("null")
}

// Retrieve variables
def instance_id = variables.get("PCA_INSTANCE_ID") as long
def instance_name = variables.get("INSTANCE_NAME")
def pca_url = variables.get('PA_CLOUD_AUTOMATION_REST_URL')
def pa_node_name = variables.get("PA_NODE_NAME")
def pa_node_host = variables.get("PA_NODE_HOST")
def pa_node_source = variables.get("PA_NODE_SOURCE")
def pa_node_url = variables.get("PA_NODE_URL")
def pa_job_id = variables.get("PA_JOB_ID")

try {
    // Acquire session id (SCHEDULER API)
    schedulerapi.connect()
    def session_id = schedulerapi.getSession()

    // To communicate with the service instance (SERVICE INSTANCE REST API)
    def service_instance_rest_api = new ServiceInstanceRestApi(new ApiClient().setBasePath(pca_url))

    // Get the service instance data (SERVICE INSTANCE REST API)
    def service_instance_data = service_instance_rest_api.getServiceInstanceUsingGET(session_id, instance_id)

    if(is_defined(endpoint_id)) {

        def deployment = new Deployment()

        // Node
        def node = new Node();
        node.setName(pa_node_name)
        node.setHost(pa_node_host)
        node.setNodeSourceName(pa_node_source)
        node.setUrl(pa_node_url)
        deployment.setNode(node)

        println "[Update_service_instance_and_add_token] adding a new deployment with pa_node_name " + pa_node_name

        // Endpoint
        def endpoint = new Endpoint();
        endpoint.setId(endpoint_id);
        endpoint.setUrl(endpoint_url);

        // Endpoint Proxy settings
        if (is_defined(proxyfied_url)) {
            endpoint.setProxyfied(true);
            endpoint.setProxyfiedUrl(proxyfied_url)
            service_instance_data = service_instance_data.addGroupsItem("scheduleradmins")
            service_instance_data = service_instance_data.addGroupsItem("rmcoreadmins")
        } else {
            endpoint.setProxyfied(false)
        }
        deployment.setEndpoint(endpoint)

        // Container
        if(is_defined(container_id)) {
            def container = new Container()
            container.setId(container_id)
            container.setName(container_name)
            deployment.setContainer(container)
        }

        service_instance_data = service_instance_data.addDeploymentsItem(deployment)
    }

    // Set the service instance status
    if (is_defined(service_instance_status)){
        service_instance_data.setInstanceStatus(service_instance_status)
    }

    // Update service instance (SERVICE INSTANCE REST API)
    service_instance_rest_api.updateServiceInstanceUsingPUT(session_id, instance_id, service_instance_data)

    // Register the service (SCHEDULER API)
    if (register_service) {
        schedulerapi.registerService(pa_job_id, instance_id as int, true)
    }

    // Add token to the current node (RM API)
    if (is_defined(token_name)){
        rmapi.connect()
        println "[Update_service_instance_and_add_token] adding token " + token_name + " to node " + pa_node_url
        rmapi.addNodeToken(pa_node_url, token_name)
    }

    // Inform other platforms that service is RUNNING (SYNCHRONIZATION API)
    if (inform_service_running) {
        def channel = "Service_Instance_" + instance_id
        synchronizationapi.createChannelIfAbsent(channel, true)
        synchronizationapi.put(channel, "RUNNING", true)
        synchronizationapi.put(channel, "INSTANCE_NAME", instance_name)
    }

} catch (Exception e) {
    StackTraceUtils.printSanitizedStackTrace(e)
    throw e
}