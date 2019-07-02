import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.pca.service.client.model.ServiceInstanceData
import org.ow2.proactive.pca.service.client.model.ServiceDescription
import org.ow2.proactive.pca.service.client.model.CloudAutomationWorkflow
import org.ow2.proactive.pca.service.client.api.CatalogRestApi

println("BEGIN " + variables.get("PA_TASK_NAME"))

// Get schedulerapi access
schedulerapi.connect()

// Acquire session id
def sessionId = schedulerapi.getSession()

// Define PCA URL
def scheduler_rest_url = variables.get("PA_SCHEDULER_REST_URL")
def pcaUrl = scheduler_rest_url.replaceAll("/rest\\z", "/cloud-automation-service")

// Connect to APIs
def apiClient = new ApiClient()
apiClient.setBasePath(pcaUrl)
//apiClient.setDebugging(true)
def serviceInstanceRestApi = new ServiceInstanceRestApi(apiClient)

def serviceId = variables.get("SERVICE_ID")
def instanceName = variables.get("INSTANCE_NAME")

//Check that the provided serviceId belongs to the existing Service Activation list
def catalogRestApi = new CatalogRestApi(apiClient)
if(!catalogRestApi.listAllWorkflowsByServiceIdUsingGET(sessionId).keySet().contains(serviceId)){
    throw new IllegalArgumentException("The provided SERVICE_ID:<" + serviceId + "> does not belong to the existing Service Activation list. You have to specify an existing service id.")
}

println("SERVICE_ID:    " + serviceId)
println("INSTANCE_NAME: " + instanceName)

def startingState = variables.get("STARTING_STATE")
if (!startingState) {
    startingState = "RUNNING"
}

// Check existing service instances
boolean instance_exists = false
List<ServiceInstanceData> service_instances = serviceInstanceRestApi.getServiceInstancesUsingGET()
for (ServiceInstanceData serviceInstanceData : service_instances) {
    if ( (serviceInstanceData.getServiceId() == serviceId) && (serviceInstanceData.getInstanceStatus()  == startingState)){
        if (serviceInstanceData.getVariables().get("INSTANCE_NAME") == instanceName) {
            instance_exists = true
            def instanceId = serviceInstanceData.getInstanceId()
            endpoint = serviceInstanceData.getInstanceEndpoints().entrySet().iterator().next().getValue()
            println("INSTANCE_ID: " + instanceId)
            println("ENDPOINT:    " + endpoint)
            variables.put("INSTANCE_ID_" + instanceName, instanceId)
            variables.put("ENDPOINT_" + instanceName, endpoint)
            result = endpoint
            break
        }
    }
}

if (!instance_exists){
    //Identifying the starting workflow, the bucket name and the default variables inside the catalog
    def startingWorkflowName
    def bucketName
    def serviceVariables = new HashMap()
    Map<String, List<CloudAutomationWorkflow>> listStartingWorkflowsByServiceId = catalogRestApi.listStartingWorkflowsByServiceIdUsingGET(sessionId)
    for(String serviceIdIterator : listStartingWorkflowsByServiceId.keySet()){
        if (serviceIdIterator.equals(serviceId)){
            startingWorkflowName = listStartingWorkflowsByServiceId.get(serviceIdIterator)[0].getName()
            bucketName = listStartingWorkflowsByServiceId.get(serviceIdIterator)[0].getBucket()
            serviceVariables = listStartingWorkflowsByServiceId.get(serviceIdIterator)[0].getVariables().collectEntries {var -> [var.getName(), var.getValue()]}
            break
        }
    }

    // Retrieve and update workflow variables
    if (binding.variables["args"]){
        for (String var: args){
            serviceVariables.put(var, variables.get(var))
        }
    }

    // Prepare service description
    ServiceDescription serviceDescription = new ServiceDescription()
    serviceDescription.setBucketName(bucketName)
    serviceDescription.setWorkflowName(startingWorkflowName)
    if( !serviceVariables.isEmpty() ){
        serviceVariables.each{ k, v -> serviceDescription.putVariablesItem("${k}", "${v}") }
    }
    // Add INSTANCE_NAME variable which is conventionnally required for docker-based PCA Services
    serviceDescription.putVariablesItem("INSTANCE_NAME", instanceName)

    // Run service
    def serviceInstanceData = serviceInstanceRestApi.createRunningServiceInstanceUsingPOST(sessionId, serviceDescription)

    // Acquire service Instance ID
    def serviceInstanceId = serviceInstanceData.getInstanceId()

    // Create synchro channel
    def channel = "Service_Instance_" + serviceInstanceId
    println("CHANNEL: " + channel)
    synchronizationapi.createChannelIfAbsent(channel, false)
    synchronizationapi.waitUntil(channel, startingState, "{k,x -> x == true}")

    // Acquire service endpoint
    serviceInstanceData = serviceInstanceRestApi.getServiceInstanceUsingGET(serviceInstanceId)
    def instanceId = serviceInstanceData.getInstanceId()
    endpoint = serviceInstanceData.getInstanceEndpoints().entrySet().iterator().next().getValue()

    println("INSTANCE_ID: " + instanceId)
    println("ENDPOINT: " + endpoint)

    variables.put("INSTANCE_ID_" + instanceName, instanceId)
    variables.put("ENDPOINT_" + instanceName, endpoint)
    result = endpoint
}

println("END " + variables.get("PA_TASK_NAME"))