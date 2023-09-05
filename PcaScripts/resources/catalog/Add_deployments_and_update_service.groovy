import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.pca.service.client.model.Deployment
import org.ow2.proactive.pca.service.client.model.Container
import org.ow2.proactive.pca.service.client.model.Endpoint
import org.ow2.proactive.pca.service.client.model.Node
import groovy.json.JsonSlurper

if (args.length != 2) {
    println("[Add_deployments_and_update_service] ERROR: Number of arguments must be == 2")
    throw new IllegalArgumentException("Number of arguments must be == 2")
}

// Retrieve script arguments
def deployment_json_variable_basename = args[0]
def nb_deployments = args[1] as Integer

// Retrieve variables
def instance_id = variables.get("PCA_INSTANCE_ID") as long
def pa_job_id = variables.get("PA_JOB_ID")
def pca_url = variables.get('PA_CLOUD_AUTOMATION_REST_URL')
def instance_name = variables.get("INSTANCE_NAME")

// Acquire session id (SCHEDULER API)
schedulerapi.connect()
def session_id = schedulerapi.getSession()

// To communicate with the service instance (SERVICE INSTANCE REST API)
def service_instance_rest_api = new ServiceInstanceRestApi(new ApiClient().setBasePath(pca_url))

// Get the service instance data (SERVICE INSTANCE REST API)
def service_instance_data = service_instance_rest_api.getServiceInstanceUsingGET(session_id, instance_id)


// Iterate over deployments
def slurper = new JsonSlurper()
for (i = 0; i < nb_deployments; i++) {

    def current_deployment_json = variables.get(deployment_json_variable_basename + i)
    def current_deployment_map = (Map) slurper.parseText(current_deployment_json)

    // Cannot create a deployment without an endpoint
    if(current_deployment_map.containsKey("endpoint")) {

        def deployment = new Deployment()

        // Endpoint
        def endpoint = new Endpoint()
        endpoint.setId(current_deployment_map.endpoint.id)
        if(current_deployment_map.endpoint.containsKey("url")){
            endpoint.setUrl(current_deployment_map.endpoint.url)
        }

        // Endpoint (Proxy settings)
        if (current_deployment_map.endpoint.containsKey("proxyfied_url")) {
            endpoint.setProxyfied(true)
            endpoint.setProxyfiedUrl(current_deployment_map.endpoint.proxyfied_url)
            service_instance_data = service_instance_data.addGroupsItem("scheduleradmins")
            service_instance_data = service_instance_data.addGroupsItem("rmcoreadmins")
        } else {
            endpoint.setProxyfied(false)
        }
        deployment.setEndpoint(endpoint)

        // Node
        if(current_deployment_map.containsKey("node")){
            def node = new Node();
            node.setName(current_deployment_map.node.name)
            node.setHost(current_deployment_map.node.host)
            node.setNodeSourceName(current_deployment_map.node.node_source_name)
            node.setUrl(current_deployment_map.node.url)
            deployment.setNode(node)
        }

        // Container
        if(current_deployment_map.containsKey("container")){
            def container = new Container()
            container.setId(current_deployment_map.container.id)
            container.setName(current_deployment_map.container.name)
            deployment.setContainer(container)
        }

        println "[Add_deployments_and_update_service] Adding deployment " + deployment
        service_instance_data = service_instance_data.addDeploymentsItem(deployment)
    }
}

if (nb_deployments > 0) {

    // Set the service instance status
    service_instance_data.setInstanceStatus("RUNNING")

    // Update service instance (SERVICE INSTANCE REST API)
    service_instance_rest_api.updateServiceInstanceUsingPUT(session_id, instance_id, service_instance_data)

    // Register the service (SCHEDULER API)
    schedulerapi.registerService(pa_job_id, instance_id as int, true)

    // Inform other platforms that service is RUNNING (SYNCHRONIZATION API)
    def channel = "Service_Instance_" + instance_id
    synchronizationapi.createChannelIfAbsent(channel, true)
    synchronizationapi.put(channel, "RUNNING_STATE", 1)
    synchronizationapi.put(channel, "INSTANCE_NAME", instance_name)

} else {

    // Set the service instance status
    service_instance_data.setInstanceStatus("ERROR")

    // Update service instance (SERVICE INSTANCE REST API)
    service_instance_rest_api.updateServiceInstanceUsingPUT(session_id, instance_id, service_instance_data)

    // Ensure the service is not registered (SCHEDULER API)
    schedulerapi.registerService(pa_job_id, instance_id as int, false)

    // Inform other platforms that service is in ERROR (SYNCHRONIZATION API)
    def channel = "Service_Instance_" + instance_id
    synchronizationapi.createChannelIfAbsent(channel, true)
    synchronizationapi.put(channel, "RUNNING_STATE", 2)
    synchronizationapi.put(channel, "DEPLOY_ERROR_MESSAGE", variables.get("DEPLOY_ERROR_MESSAGE"))
    synchronizationapi.put(channel, "INSTANCE_NAME", instance_name)

}