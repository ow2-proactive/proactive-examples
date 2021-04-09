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

def i = 0
endpoint_id = arguments_array[i++].trim()
endpoint_url = arguments_array[i++].trim()
container_id = arguments_array[i++].trim()
container_name = arguments_array[i++].trim()
proxyfied_url = arguments_array[i++].trim()
service_instance_status = arguments_array[i++].trim()
register_service = arguments_array[i++].trim()
token_name = arguments_array[i++].trim()

println "From Update_service_instance_and_add_token"
println "endpoint_id " + endpoint_id
println "endpoint_url " + endpoint_url
println "container_id " + container_id
println "container_name " + container_name
println "proxyfied_url " + proxyfied_url
println "service_instance_status " + service_instance_status
println "register_service " + register_service
println "token_name " + token_name

// Retrieve variables
def instance_id = variables.get("PCA_INSTANCE_ID") as long
def pca_url = variables.get('PA_CLOUD_AUTOMATION_REST_URL')
def pa_node_name = variables.get("PA_NODE_NAME")
def pa_node_host = variables.get("PA_NODE_HOST")
def pa_node_source = variables.get("PA_NODE_SOURCE")
def pa_node_url = variables.get("PA_NODE_URL")
def pa_job_id = variables.get("PA_JOB_ID")

try {
    // Get schedulerapi access and acquire session id
    schedulerapi.connect()
    def session_id = schedulerapi.getSession()

    // Connect to Cloud Automation API
    def service_instance_rest_api = new ServiceInstanceRestApi(new ApiClient().setBasePath(pca_url))

    // Get service instance
    def service_instance_data = service_instance_rest_api.getServiceInstanceUsingGET(session_id, instance_id)

    if(!Strings.isNullOrEmpty(endpoint_id)) {

        println "Adding a new deployment"
        def deployment = new Deployment()

        // Node
        def node = new Node();
        node.setName(pa_node_name)
        node.setHost(pa_node_host)
        node.setNodeSourceName(pa_node_source)
        node.setUrl(pa_node_url)
        deployment.setNode(node)

        // Endpoint
        def endpoint = new Endpoint();
        endpoint.setId(endpoint_id);
        endpoint.setUrl(endpoint_url);

        // Endpoint Proxy settings
        if (!Strings.isNullOrEmpty(proxyfied_url)) {
            endpoint.setProxyfied(true);
            endpoint.setProxyfiedUrl(proxyfied_url)
            service_instance_data = service_instance_data.addGroupsItem("scheduleradmins")
            service_instance_data = service_instance_data.addGroupsItem("rmcoreadmins")
        } else {
            endpoint.setProxyfied(false)
        }
        deployment.setEndpoint(endpoint)

        // Container
        if(!Strings.isNullOrEmpty(container_id)) {
            def container = new Container()
            container.setId(container_id)
            container.setName(container_name)
            deployment.setContainer(container)
        }

        service_instance_data = service_instance_data.addDeploymentsItem(deployment)
    }

    // Set the service instance status
    if (!Strings.isNullOrEmpty(service_instance_status)){
        service_instance_data.setInstanceStatus(service_instance_status)
    }

    // Update service instance
    service_instance_data = service_instance_rest_api.updateServiceInstanceUsingPUT(session_id, instance_id, service_instance_data)
    println(service_instance_data)

    // Register the service
    if (register_service.toLowerCase()=="true") {
        schedulerapi.registerService(pa_job_id, instance_id as int, true)
    }

    // Add token to the current node
    if (!Strings.isNullOrEmpty(token_name)){
        rmapi.connect()
        println "ADDING TOKEN " + token_name + " TO NODE " + pa_node_url
        rmapi.addNodeToken(pa_node_url, token_name)
    }

} catch (Exception e) {
    StackTraceUtils.printSanitizedStackTrace(e)
    throw e
}