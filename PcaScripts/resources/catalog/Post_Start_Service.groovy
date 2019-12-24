/*********************************************************************************
* THIS POSTSCRIPT PROPAGATES USEFUL INFORMATION SUCH AS:                         *
* 1) SERVICE ENDPOINT (PROTOCOL://HOSTNAME:PORT)                                 *
* 2) SERVICE STATUS "RUNNING" VIA SYNCHRONIZATION API      						 *
*********************************************************************************/

import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi

def paSchedulerRestUrl = variables.get('PA_SCHEDULER_REST_URL')
def pcaUrl = paSchedulerRestUrl.replaceAll("/rest\\z", "/cloud-automation-service")
// Acquire service instance id and instance name
def instanceId = variables.get("PCA_INSTANCE_ID") as long
def instanceName = variables.get("INSTANCE_NAME")

def hostname = new URL(paSchedulerRestUrl).getHost()
def port = new File(instanceName+"_port").text.trim()
def endpoint = 'http://'+ hostname + ":" + port
variables.put("HOSTNAME", hostname)
variables.put("PORT", port)

// Connect to Cloud Automation API
def apiClient = new ApiClient()
apiClient.setBasePath(pcaUrl)
def serviceInstanceRestApi = new ServiceInstanceRestApi(apiClient)

//Update endpoints
def endpointsMap = [:]
endpointsMap.put(instanceName, endpoint)
serviceInstanceRestApi.createNewInstanceEndpointsUsingPUT(instanceId, endpointsMap)

// Update service instance data : (status, endpoint)
def serviceInstanceData = serviceInstanceRestApi.getServiceInstanceUsingGET(instanceId)
def status = new File(instanceName+"_status").text.trim()
def currentStatus = (!status.equals(instanceName)) ? "ERROR" : "RUNNING"
serviceInstanceData.setInstanceStatus(currentStatus)
serviceInstanceData = serviceInstanceRestApi.updateServiceInstanceUsingPUT(instanceId, serviceInstanceData)

// Print warning or error messages and force job to exit with error if there are any.
if (!status.equals(instanceName)){
    println("[ERROR] Could not start docker container: " + instanceName + ". Docker output: " + status)
    System.exit(1)
}

// Inform other platforms that service is running through Synchronization API
def channel = "Service_Instance_" + instanceId
synchronizationapi.createChannelIfAbsent(channel, false)
synchronizationapi.put(channel, "RUNNING", true)
synchronizationapi.put(channel, "INSTANCE_NAME", instanceName)

// Add token to the current node
token = instanceName
nodeUrl = variables.get("PA_NODE_URL")
println("Current nodeUrl: " + nodeUrl)
println("Adding token:    " + token)
rmapi.connect()
rmapi.addNodeToken(nodeUrl, token)


// LOG OUTPUT
println(variables.get("PA_JOB_NAME") + "_INSTANCE_ID: " + instanceId)
println(variables.get("PA_JOB_NAME") + "_ENDPOINT: " + endpoint)

println("END " + variables.get("PA_TASK_NAME"))