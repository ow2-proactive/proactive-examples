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
def pcaUrl = variables.get('PA_CLOUD_AUTOMATION_REST_URL')

// Connect to APIs
def apiClient = new ApiClient()
apiClient.setBasePath(pcaUrl)
//apiClient.setDebugging(true)
def serviceInstanceRestApi = new ServiceInstanceRestApi(apiClient)

def serviceId = variables.get("SERVICE_ID")
def instanceName = variables.get("INSTANCE_NAME")

def publishService = false
def enableServiceActions = true
if (binding.variables["args"]) {
    if (args.length > 0) {
        try {
            publishService = Boolean.parseBoolean(args[0])
        } catch (Exception e) {
            println "Invalid first argument, expected boolean, received " + args[0]
        }
    }
    if (args.length > 1) {
        try {
            enableServiceActions = Boolean.parseBoolean(args[1])
        } catch (Exception e) {
            println "Invalid second argument, expected boolean, received " + args[1]
        }
    }
}


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
List<ServiceInstanceData> service_instances = serviceInstanceRestApi.getServiceInstancesUsingGET(sessionId)
for (ServiceInstanceData serviceInstanceData : service_instances) {
    if ( (serviceInstanceData.getServiceId() == serviceId) && (serviceInstanceData.getInstanceStatus()  == startingState)){
        if (serviceInstanceData.getVariables().get("INSTANCE_NAME") == instanceName) {
            instance_exists = true
            def instanceId = serviceInstanceData.getInstanceId()
            endpoint = serviceInstanceData.getDeployments().iterator().next().getEndpoint().getUrl()
            println("INSTANCE_ID: " + instanceId)
            println("ENDPOINT:    " + endpoint)
            variables.put("INSTANCE_ID_" + instanceName, instanceId)
            variables.put("ENDPOINT_" + instanceName, endpoint)
            if (publishService) {
                schedulerapi.registerService(variables.get("PA_JOB_ID"), instanceId as int, enableServiceActions)
            }
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
            if (variables.containsKey(var)) {
                serviceVariables.put(var, variables.get(var))
            }
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
    def serviceInstanceData = serviceInstanceRestApi.createRunningServiceInstanceUsingPOST(sessionId, serviceDescription, variables.get("PA_JOB_ID"))

    // Acquire service Instance ID
    def serviceInstanceId = serviceInstanceData.getInstanceId()

    // Create synchro channel
    def channel = "Service_Instance_" + serviceInstanceId
    println("CHANNEL: " + channel)
    synchronizationapi.createChannelIfAbsent(channel, false)
    synchronizationapi.waitUntil(channel, startingState, "{k,x -> x == true}")

    // Acquire service endpoint
    serviceInstanceData = serviceInstanceRestApi.getServiceInstanceUsingGET(sessionId, serviceInstanceId)
    def instanceId = serviceInstanceData.getInstanceId()
    endpoint = serviceInstanceData.getDeployments().iterator().next().getEndpoint().getUrl()

    if (publishService) {
        schedulerapi.registerService(variables.get("PA_JOB_ID"), instanceId as int, enableServiceActions)
    }

    println("INSTANCE_ID: " + instanceId)
    println("ENDPOINT: " + endpoint)

    variables.put("INSTANCE_ID_" + instanceName, instanceId)
    variables.put("ENDPOINT_" + instanceName, endpoint)
    result = endpoint
}

println("END " + variables.get("PA_TASK_NAME"))