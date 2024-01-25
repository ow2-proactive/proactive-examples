///////////////////////////////////////////////////////////////////////////////
//   This script is used to start a PSA service from a Groovy task           //
//   To use this script in a task, users should provide:                     //
//   1- A task variable named `INSTANCE_NAME` of type PA:NOT_EMPTY_STRING    //
//   that will be used to identify the PSA instance                          //
//   2- At least one task variable to identify the service to deploy:        //
//      i- A valid `SERVICE_ID` of type PA:NOT_EMPTY_STRING                  //
//     ii- A `SERVICE_ACTIVATION_WORKFLOW` of type PA:CATALOG_OBJECT         //
//    N.B: We recommend using `SERVICE_ACTIVATION_WORKFLOW`.                 //
//   3- A boolean argument (true/false) to say if the related service        //
//      should be published.                                                 //
///////////////////////////////////////////////////////////////////////////////

import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.pca.service.client.model.ServiceInstanceData
import org.ow2.proactive.pca.service.client.model.ServiceDescription
import org.ow2.proactive.pca.service.client.model.CloudAutomationWorkflow
import org.ow2.proactive.pca.service.client.api.CatalogRestApi
import com.google.common.base.Strings;

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
def catalogRestApi = new CatalogRestApi(apiClient)

def serviceId = variables.get("SERVICE_ID")
def instanceName = variables.get("INSTANCE_NAME")
def serviceActivationWorkflow = variables.get("SERVICE_ACTIVATION_WORKFLOW")

def bucketName
def startingWorkflowName
def serviceVariables

if (serviceActivationWorkflow != null) {
    def serviceActivationWorkflowSplits = serviceActivationWorkflow.split('/')
    bucketName = serviceActivationWorkflowSplits[0]
    startingWorkflowName = serviceActivationWorkflowSplits[1]
    println("Service Bucket_name: " + bucketName + ", Workflow_name: " + startingWorkflowName)
}

def publishService = false
def enableServiceActions = true
if (binding.variables["args"]) {
    if (args.length > 0) {
        publishService = Boolean.parseBoolean(args[0])
    }
    if (args.length > 1) {
        enableServiceActions = Boolean.parseBoolean(args[1])
    }
}

if (!Strings.isNullOrEmpty(serviceId)) {
    // Get the service workflow
    // Check that the provided serviceId belongs to the existing Service Activation list
    List<CloudAutomationWorkflow> listStartingWorkflows = catalogRestApi.getWorkflowsByServiceIdUsingGET(sessionId, serviceId)
    if (!listStartingWorkflows) {
        throw new IllegalArgumentException("The provided SERVICE_ID is not valid.")
    }
    // Check when SERVICE_ID and SERVICE_ACTIVATION_WORKFLOW are provided the specific service activation workflow
    if (serviceActivationWorkflow != null) {
        for (CloudAutomationWorkflow cloudAutomationWorkflow : listStartingWorkflows) {
            if (cloudAutomationWorkflow.getName().equals(startingWorkflowName) && cloudAutomationWorkflow.getBucket().equals(bucketName)) {
                serviceVariables = cloudAutomationWorkflow.getVariables().collectEntries {var -> [var.getName(), var.getValue()]}
                println("The provided SERVICE_ID and SERVICE_ACTIVATION_WORKFLOW are compatible.")
                break
            }
        }
        if (serviceVariables == null) {
            throw new IllegalArgumentException("The provided SERVICE_ID and SERVICE_ACTIVATION_WORKFLOW does not belong to the same PSA Service.")
        }
    } else {
        // Get the first valid activation workflow
        startingWorkflowName = listStartingWorkflows[0].getName()
        bucketName = listStartingWorkflows[0].getBucket()
        serviceVariables = listStartingWorkflows[0].getVariables().collectEntries {var -> [var.getName(), var.getValue()]}
    }
} else if (serviceActivationWorkflow != null) {
    //Identifying the starting workflow, the service ID and the default variables inside the catalog
    def cloudAutomationWorkflow = catalogRestApi.getWorkflowByCatalogObjectUsingGET(sessionId, bucketName, startingWorkflowName)
    if (cloudAutomationWorkflow != null) {
        serviceId = cloudAutomationWorkflow.getGenericInformation().getServiceId()
        serviceVariables = cloudAutomationWorkflow.getVariables().collectEntries {var -> [var.getName(), var.getValue()]}
        println("Found Service_id : " + serviceId)
    }
} else {
    throw new IllegalArgumentException("The provided SERVICE_ID or SERVICE_ACTIVATION_WORKFLOW does not belong to the existing Service Activation list. You have to specify an existing service ID or service activation workflow.")
}

println("SERVICE_ID: " + serviceId)
println("INSTANCE_NAME: " + instanceName)

// Check existing service instances
boolean instance_exists = false
List<ServiceInstanceData> service_instances = serviceInstanceRestApi.getServiceInstancesUsingGET(sessionId, null)
for (ServiceInstanceData serviceInstanceData : service_instances) {
    if ( (serviceInstanceData.getServiceId() == serviceId) && (serviceInstanceData.getInstanceStatus()  == "RUNNING")){
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

    // Wait until the service is RUNNING or in ERROR
    synchronizationapi.waitUntil(channel, "RUNNING_STATE", "{k,x -> x > 0}")
    def runningState = synchronizationapi.get(channel, "RUNNING_STATE") as int

    // If RUNNING
    if (runningState == 1) {

        // Acquire service endpoint
        serviceInstanceData = serviceInstanceRestApi.getServiceInstanceUsingGET(sessionId, serviceInstanceId)
        endpoint = serviceInstanceData.getDeployments().iterator().next().getEndpoint().getUrl()

        // Acquire service job id
        serviceJobId = serviceInstanceData.getJobSubmissions().get(0).getJobId().toString()
        variables.put("SERVICE_JOB_ID", serviceJobId)
        println("SERVICE_JOB_ID: " + serviceJobId)

        if (publishService) {
            schedulerapi.registerService(variables.get("PA_JOB_ID"), serviceInstanceId as int, enableServiceActions)
        }

        println("INSTANCE_ID: " + serviceInstanceId)
        println("ENDPOINT: " + endpoint)
        variables.put("INSTANCE_ID_" + instanceName, serviceInstanceId)
        variables.put("ENDPOINT_" + instanceName, endpoint)
        result = endpoint

        // If in ERROR
    } else if (runningState == 2) {

        // Make the task in error
        def deployErrorMessage = synchronizationapi.get(channel, "DEPLOY_ERROR_MESSAGE")
        println deployErrorMessage
        throw new Exception()
    }
}

println("END " + variables.get("PA_TASK_NAME"))