import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi

// Arguments must be var0 val0 var1 val1 (if val == VARIABLE_VALUE, let's consider var == val)
if (args.length < 2 && (args.length % 2) != 0) {
    println("[Propagate_variables_to_current_service] ERROR Number of arguments must be even and > 1")
    throw new IllegalArgumentException("Number of arguments must be even and > 1")
}

// Retrieve variables
def pca_url = variables.get('PA_CLOUD_AUTOMATION_REST_URL')
def instance_id = variables.get("PCA_INSTANCE_ID") as long

// Get schedulerapi access and acquire session id
schedulerapi.connect()
def session_id = schedulerapi.getSession()

// Connect to APIs
def service_instance_rest_api = new ServiceInstanceRestApi(new ApiClient().setBasePath(pca_url))

// Update the service instance
def service_instance_data = service_instance_rest_api.getServiceInstance(session_id, instance_id)
for  (i = 0; i < args.length-1; i = i+2) {
    if ("VARIABLE_VALUE".equals(args[i+1])) {
        service_instance_data.getVariables().put(args[i], variables.containsKey(args[i]) ? variables.get(args[i]) : "" )
    } else {
        service_instance_data.getVariables().put(args[i], args[i+1])
    }
}
service_instance_rest_api.updateServiceInstance(session_id, instance_id, service_instance_data)