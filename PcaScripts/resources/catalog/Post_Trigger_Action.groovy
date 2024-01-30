/*********************************************************************************
* THIS POSTSCRIPT INFORMS PLATFORM THAT PCA SERVICE ACION IS TRIGGERED                   *
*********************************************************************************/

import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.pca.service.client.model.ServiceInstanceData

def action = args[0]

def pcaUrl = variables.get('PA_CLOUD_AUTOMATION_REST_URL')
def instanceId = variables.get("PCA_INSTANCE_ID") as long
def instanceName = variables.get("INSTANCE_NAME")

def ALREADY_REMOVED_MESSAGE = "Error: No such container: " + instanceName

// Get schedulerapi access and acquire session id
schedulerapi.connect()
def sessionId = schedulerapi.getSession()

// Connect to Cloud Automation API
def serviceInstanceRestApi = new ServiceInstanceRestApi(new ApiClient().setBasePath(pcaUrl))

// Update service instance data : (status, endpoint)
def status = new File(instanceName+"_status").text.trim()
def currentStatus = (!status.equals(ALREADY_REMOVED_MESSAGE) && !status.equals(instanceName)) ? "ERROR" : action
def serviceInstanceData = serviceInstanceRestApi.getServiceInstance(sessionId, instanceId)
serviceInstanceData.setInstanceStatus(currentStatus)
serviceInstanceRestApi.updateServiceInstance(sessionId, instanceId, serviceInstanceData)

if(action.equals("FINISHED")){
    // Inform other jobs that the service is finished and deleted.
	def channel = "Service_Instance_" + instanceId
	synchronizationapi.put(channel, "FINISH_DONE", true)
}

// Print warning or error messages and force job to exit with error if there are any.
if (status.equals(ALREADY_REMOVED_MESSAGE)){
    println("[WARNING] docker container: " + instanceName + " is already removed.")
} else if (!status.equals(instanceName)){
    println("[ERROR] Could not remove docker container: " + instanceName + ". Docker output: " + status)
    throw new IllegalStateException("Could not remove docker container: " + instanceName + ". Docker output: " + status)
}

println("END " + variables.get("PA_TASK_NAME"))