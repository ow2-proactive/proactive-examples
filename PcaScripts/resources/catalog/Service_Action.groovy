import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.CatalogRestApi
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.pca.service.client.model.CloudAutomationWorkflow
import org.ow2.proactive.pca.service.client.model.ServiceDescription
import java.util.concurrent.TimeoutException
import com.google.common.base.Strings

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
def instanceName = variables.get("INSTANCE_NAME")
def instanceId = (!variables.get("INSTANCE_ID") && instanceName)? variables.get("INSTANCE_ID_" + instanceName) : variables.get("INSTANCE_ID")
if (!instanceId && !instanceName){
    throw new IllegalArgumentException("You have to specify an INSTANCE_NAME or an INSTANCE_ID. Empty value for both is not allowed.");
}

println("INSTANCE_ID: " + instanceId)

def bucketName
def isActionExists = false
def catalogRestApi = new CatalogRestApi(apiClient)
def actionVariables = new HashMap()

def action = variables.get("ACTION")
def serviceActionWorkflow = variables.get("SERVICE_ACTION_WORKFLOW")

if (Strings.isNullOrEmpty(action) && Strings.isNullOrEmpty(serviceActionWorkflow)) {
    throw new IllegalArgumentException("You have to provide an ACTION value or a SERVICE_ACTION_WORKFLOW. Empty value is not allowed.");
}

if (!Strings.isNullOrEmpty(action)) {
    List<CloudAutomationWorkflow> listExecutableActions = catalogRestApi.listExecutableActionsByInstanceId(sessionId, instanceId.toString()).get(instanceId.toString())
    for (CloudAutomationWorkflow actionIterator : listExecutableActions) {
        if (actionIterator.getName().equals(action)){
            bucketName = actionIterator.getBucket()
            //retrieve default action variables
            actionVariables = actionIterator.getVariables().collectEntries {var -> [var.getName(), var.getValue()]}
            isActionExists = true
            break
        }
    }
} else if (!Strings.isNullOrEmpty(serviceActionWorkflow)) {
    def serviceActionWorkflowSplits = serviceActionWorkflow.split('/')
    bucketName = serviceActionWorkflowSplits[0]
    action = serviceActionWorkflowSplits[1]
    CloudAutomationWorkflow executableAction = catalogRestApi.getExecutableActionByCatalogObject(sessionId, instanceId as int, bucketName, action)
    println("Action Bucket_name: " + bucketName + ", Action_workflow_name: " + action)
    if (executableAction != null) {
        actionVariables = executableAction.getVariables().collectEntries {var -> [var.getName(), var.getValue()]}
        isActionExists = true
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
def serviceInstanceData = serviceInstanceRestApi.launchServiceInstanceAction(sessionId, instanceId as int, service, variables.get("PA_JOB_ID"))

if (action.toLowerCase().contains("finish")) {
    try {
        serviceJobId = serviceInstanceData.getJobSubmissions().get(0).getJobId().toString()
        println("SERVICE_JOB_ID: " + serviceJobId)
        schedulerapi.waitForJob(serviceJobId, 180000)
    } catch (TimeoutException toe) {
        println("[Warning] Timeout reached. Disable to wait until the PCA service " + instanceId + " finishes." )
    }
}

println("END " + variables.get("PA_TASK_NAME"))
