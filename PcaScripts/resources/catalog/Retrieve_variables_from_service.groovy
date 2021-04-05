import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi

// Arguments must be var0 var1 ...
if (args.length < 1) {
    println("[ERROR] Number of arguments must be > 0")
    System.exit(1)
}

// Retrieve variables
def pca_url = variables.get('PA_CLOUD_AUTOMATION_REST_URL')
def service_instance_id = variables.get("service_instance_id") as long

// Get schedulerapi access and acquire session id
schedulerapi.connect()
def session_id = schedulerapi.getSession()

// Connect to APIs
def service_instance_rest_api = new ServiceInstanceRestApi(new ApiClient().setBasePath(pca_url))

// Retrieve service variables
def service_instance_data = service_instance_rest_api.getServiceInstanceUsingGET(session_id, service_instance_id)
for  (i = 0; i < args.length; i++) {
    variables.put(args[i], service_instance_data.getVariables().get(args[i]))
}