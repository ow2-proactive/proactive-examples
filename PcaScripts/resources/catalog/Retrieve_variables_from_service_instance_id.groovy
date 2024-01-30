import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.scheduler.common.util.VariableSubstitutor

// Arguments must be <service_id> <var0_name_to_store> <var0_to_retrieve> <var1_name_to_store> <var1_to_retrieve> ...
if (args.length < 3) {
    println "[Retrieve_variables_from_service_instance_id] ERROR: Number of arguments must be >= 3"
    throw new IllegalArgumentException("Number of arguments must be >= 3")
}

if (args[0] == "") {
    println "[Retrieve_variables_from_service_instance_id] the service instance id is empty"
    return
}

service_instance_id = args[0] as long

// Retrieve variables
def pca_url = variables.get('PA_CLOUD_AUTOMATION_REST_URL')

// Get schedulerapi access and acquire session id
schedulerapi.connect()
def session_id = schedulerapi.getSession()

// Connect to APIs
def service_instance_rest_api = new ServiceInstanceRestApi(new ApiClient().setBasePath(pca_url))

// Retrieve service variables
def service_instance_data = service_instance_rest_api.getServiceInstance(session_id, service_instance_id)
for  (i = 1; i < args.length; i = i + 2) {
    if (args[i+1].contains("%{")) {
        // var_to_retrieve can contain a reference to another variable in the target service
        // we use the format %{var} instead of ${var} to prevent conflicts with the current workflow variables
        def modifiedPattern = args[i+1].replace("%{","\${")
        def resolvedVariable = VariableSubstitutor.filterAndUpdate(modifiedPattern, service_instance_data.getVariables())
        println "[Retrieve_variables_from_service_instance_id] Propagating " + resolvedVariable + " under " + args[i]
        variables.put(args[i], resolvedVariable)
    } else {
        println "[Retrieve_variables_from_service_instance_id] Propagating " + service_instance_data.getVariables().get(args[i+1]) + " under " + args[i]
        variables.put(args[i], service_instance_data.getVariables().get(args[i+1]))
    }
}