/*********************************************************************************
* THIS POSTSCRIPT PROPAGATES USEFUL INFORMATION SUCH AS:                         *
* 1) SERVICE ENDPOINT (PROTOCOL://HOSTNAME:PORT)                                 *
* 2) CREDENTIALS (IF THERE ARE ANY) BY ADDING THEM TO 3RD PARTY CREDENTIALS      *
*********************************************************************************/

import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.pca.service.client.model.ServiceInstanceData
import org.ow2.proactive.pca.service.client.model.Container
import org.ow2.proactive.pca.service.client.model.Endpoint
import org.ow2.proactive.pca.service.client.model.Deployment
import org.ow2.proactive.pca.service.client.model.Node
import java.net.URLEncoder;

// Acquire variables
def instanceId = variables.get("PCA_INSTANCE_ID") as long
def instanceName = variables.get("INSTANCE_NAME")
def proxyfied = variables.get("PROXYFIED")
def hostname = variables.get("PA_NODE_HOST")
def endpointID = variables.get("ENDPOINT_ID")+"-"+instanceId
def httpsEnabled = variables.get("HTTPS_ENABLED")
def userKey = variables.get("USER_KEY")
def traceEnabled = variables.get("TRACE_ENABLED")
def engine = variables.get("ENGINE")

// Handle service parameters
def port = new File(instanceName+"_port").text.trim()
def containerUrl = hostname+":"+port
def containerID = ""
if (engine != null && "singularity".equalsIgnoreCase(engine)) {
    containerID = "0"
} else {
    containerID = new File(instanceName+"_containerID").text.trim()
}

// Determine Cloud Automation URL
def pcaUrl = variables.get('PA_CLOUD_AUTOMATION_REST_URL')

// Get schedulerapi access and acquire session id
schedulerapi.connect()
def sessionId = schedulerapi.getSession()

// Connect to Cloud Automation API
def apiClient = new ApiClient()
apiClient.setBasePath(pcaUrl)
def serviceInstanceRestApi = new ServiceInstanceRestApi(apiClient)

// Implement service model

// Container
def Container container = new Container()
container.setId(containerID)
container.setName(instanceName)

// Endpoint
def Endpoint endpoint = new Endpoint();
endpoint.setId(endpointID);
// Set the endpoint parameters according to the Proxy settings
if (proxyfied.toLowerCase()=="true"){
    if (httpsEnabled.toLowerCase()=="true"){
        containerUrl = "https://"+containerUrl
    } else{
        containerUrl = "http://"+containerUrl
    }
    containerUrl = URLDecoder.decode(containerUrl, "UTF-8");
    proxyfiedURL = pcaUrl+"/services/"+instanceId+"/endpoints/"+endpointID+"/api/dashapp?key="+userKey
    endpoint.setProxyfied(true);
    endpoint.setProxyfiedUrl(proxyfiedURL)
}else{
    endpoint.setProxyfied(false)
    ENDPOINT_PATH = "/api/ui"
    if (traceEnabled.toLowerCase()=="true"){
        ENDPOINT_PATH = "/api/dashapp?key="+userKey
    }
    if (httpsEnabled.toLowerCase()=="true"){
        containerUrl = "https://"+containerUrl+ENDPOINT_PATH
    } else{
        containerUrl = "http://"+containerUrl+ENDPOINT_PATH
    }
    containerUrl = URLDecoder.decode(containerUrl, "UTF-8");
}
endpoint.setUrl(containerUrl);

// Node
def Node node = new Node();
node.setName(variables.get("PA_NODE_NAME"))
node.setHost(variables.get("PA_NODE_HOST"))
node.setNodeSourceName(variables.get("PA_NODE_SOURCE"))
node.setUrl(variables.get("PA_NODE_URL"))

// Deployment
def Deployment deployment = new Deployment()
deployment.setNode(node)
deployment.setContainer(container)
deployment.setEndpoint(endpoint)

// Update service instance model (add Deployment, Groups)
def serviceInstanceData = serviceInstanceRestApi.getServiceInstanceUsingGET(sessionId, instanceId)
serviceInstanceData.setInstanceStatus("RUNNING")
serviceInstanceData = serviceInstanceData.addDeploymentsItem(deployment)
if (proxyfied.toLowerCase()=="true"){
    serviceInstanceData = serviceInstanceData.addGroupsItem("scheduleradmins")
    serviceInstanceData = serviceInstanceData.addGroupsItem("rmcoreadmins")
}
serviceInstanceData = serviceInstanceRestApi.updateServiceInstanceUsingPUT(sessionId, instanceId, serviceInstanceData)
println(serviceInstanceData)

schedulerapi.registerService(variables.get("PA_JOB_ID"), instanceId as int, true)

// Inform other platforms that service is running through Synchronization API
channel = "Service_Instance_" + instanceId
synchronizationapi.createChannelIfAbsent(channel, false)
synchronizationapi.put(channel, "RUNNING_STATE", 1)
synchronizationapi.put(channel, "INSTANCE_NAME", instanceName)

// Add token to the current node
token = "PSA_" + instanceName
nodeUrl = variables.get("PA_NODE_URL")
println("Current nodeUrl: " + nodeUrl)
println("Adding token:    " + token)
rmapi.connect()
rmapi.addNodeToken(nodeUrl, token)

// Log output
println(variables.get("PA_JOB_NAME") + "_INSTANCE_ID: " + instanceId)
println(variables.get("PA_JOB_NAME") + "_ENDPOINT: " + endpoint)

println("END " + variables.get("PA_TASK_NAME"))
