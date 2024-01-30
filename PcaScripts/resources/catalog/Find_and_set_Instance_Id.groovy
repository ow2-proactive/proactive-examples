import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.pca.service.client.model.ServiceInstanceData


// Get schedulerapi access
schedulerapi.connect()

// Acquire session id
def sessionId = schedulerapi.getSession()

// Define PCA URL
def pcaUrl = variables.get('PA_CLOUD_AUTOMATION_REST_URL')

// Connect to APIs
def apiClient = new ApiClient()
apiClient.setBasePath(pcaUrl)
def serviceInstanceRestApi = new ServiceInstanceRestApi(apiClient)

println("Find and set INSTANCE_ID script:")

def serviceId = args[0]
def instanceName = args[1]
println("   SERVICE_ID:    " + serviceId)
println("   INSTANCE_NAME: " + instanceName)

// Check existing service instances
List<ServiceInstanceData> service_instances = serviceInstanceRestApi.getServiceInstances(sessionId, null)

for (ServiceInstanceData serviceInstanceData : service_instances) {
    if ( serviceInstanceData.getServiceId() == serviceId ) {
        if (serviceInstanceData.getVariables().get("INSTANCE_NAME") == instanceName) {
            def instanceId = serviceInstanceData.getInstanceId()
            endpoint = serviceInstanceData.getDeployments().iterator().next().getEndpoint().getUrl()
            println("   -> INSTANCE_ID: " + instanceId)
            variables.put("INSTANCE_ID", instanceId)
            break
        }
    }
}