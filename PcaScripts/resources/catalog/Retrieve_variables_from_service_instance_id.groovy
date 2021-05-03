import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi

// Arguments must be <service_id> <var0_name_to_store> <var0_to_retrieve> <var1_name_to_store> <var1_to_retrieve> ...
if (args.length < 3) {
    println "[Retrieve_variables_from_service_instance_id] ERROR: Number of arguments must be >= 3"
    System.exit(1)
}

try {
    service_instance_id = args[0] as long
} catch (java.lang.NumberFormatException e) {
    println "[Retrieve_variables_from_service_instance_id] WARN: NumberFormatException: Invalid service_instance_id: " + args[0]
    return
}

// Retrieve variables
def pca_url = variables.get('PA_CLOUD_AUTOMATION_REST_URL')

// Get schedulerapi access and acquire session id
schedulerapi.connect()
def session_id = schedulerapi.getSession()

// Connect to APIs
def service_instance_rest_api = new ServiceInstanceRestApi(new ApiClient().setBasePath(pca_url))

// Retrieve service variables
def service_instance_data = service_instance_rest_api.getServiceInstanceUsingGET(session_id, service_instance_id)
for  (i = 1; i < args.length; i = i + 2) {
    println "[Retrieve_variables_from_service_instance_id] Propagating " + service_instance_data.getVariables().get(args[i+1]) + " under " + args[i]
    variables.put(args[i], service_instance_data.getVariables().get(args[i+1]))
}