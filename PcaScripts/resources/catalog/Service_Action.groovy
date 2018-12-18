println("BEGIN " + variables.get("PA_TASK_NAME"))

import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.pca.service.client.model.ServiceInstanceData
import org.ow2.proactive.pca.service.client.model.ServiceDescription
import org.ow2.proactive.pca.service.client.model.CloudAutomationWorkflow
import org.ow2.proactive.pca.service.client.api.CatalogRestApi

// Get schedulerapi access
schedulerapi.connect()

// Acquire session id
def sessionId = schedulerapi.getSession()

// Define PCA URL
def schedulerRestUrl = variables.get("PA_SCHEDULER_REST_URL")
def pcaUrl = schedulerRestUrl.replaceAll("/rest\\z", "/cloud-automation-service")

// Connect to APIs
def apiClient = new ApiClient()
apiClient.setBasePath(pcaUrl)
//apiClient.setDebugging(true)
def serviceInstanceRestApi = new ServiceInstanceRestApi(apiClient)
def instanceId = variables.get("INSTANCE_ID")
def instanceName = variables.get("INSTANCE_NAME")
if(instanceName){
    instanceId = variables.get("INSTANCE_ID_" + instanceName) as int
}else if (instanceId){
    instanceId = instanceId as int
}else{
    throw new IllegalArgumentException("You have to specify either an INSTANCE_NAME or an INSTANCE_ID. Both variables cannot not empty.");
}

println("INSTANCE_ID: " + instanceId)

def action = variables.get("ACTION")
if (action.isEmpty()) {
    throw new IllegalArgumentException("You have to provide an ACTION value. Empty value is not allowed.");
}

def bucketName
def isActionExists = false
def catalogRestApi = new CatalogRestApi(apiClient)
def actionVariables = new HashMap()

Map<String, List<CloudAutomationWorkflow>> listExecutableActionsByInstanceId = catalogRestApi.listExecutableActionsByInstanceIdUsingGET(sessionId)
List<CloudAutomationWorkflow> listExecutableActions = listExecutableActionsByInstanceId.get(instanceId.toString())
for (CloudAutomationWorkflow actionIterator : listExecutableActions) {
    if (actionIterator.getName().equals(action)){
        bucketName = actionIterator.getBucket()
        //retrieve default action variables
        actionVariables = actionIterator.getVariables().collectEntries {var -> [var.getName(), var.getValue()]}
        isActionExists = true
        break
    }
}
if(!isActionExists){
    throw new IllegalArgumentException("The provided ACTION: " + action + " does not belong to the existing possible actions that can be applied to the current state of the service. You have to specify a valid action.")
}

// Retrieve and update workflow variables
if (binding.variables["args"]){
    for (String var: args){
        actionVariables.put(var, variables.get(var))
    }
}

// Execute action on service
ServiceDescription service = new ServiceDescription()
service.setBucketName(bucketName)
service.setWorkflowName(action)
if( !actionVariables.isEmpty() ){
    actionVariables.each{ k, v -> service.putVariablesItem("${k}", "${v}") }
}
serviceInstanceRestApi.launchServiceInstanceActionUsingPUT(sessionId, instanceId, service)

println("END " + variables.get("PA_TASK_NAME"))
