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
import org.codehaus.groovy.runtime.StackTraceUtils

// Acquire variables
def instanceId = variables.get("PCA_INSTANCE_ID") as long
def instanceName = variables.get("INSTANCE_NAME")
def proxyfied = variables.get("PROXYFIED")
def endpointID = variables.get("ENDPOINT_ID")
def httpEnabled = variables.get("HTTP_ENABLED") // MongoDB, Cassandra, etc
def httpsEnabled = variables.get("HTTPS_ENABLED") // Visdom, Tensorboard, MaaS, JupyterLab, etc
def engine = variables.get("ENGINE") // Docker, Singularity

// Handle service parameters
def hostname = variables.get("PA_NODE_HOST")
def port = new File(instanceName+"_port").text.trim()

if (engine != null && "singularity".equalsIgnoreCase(engine)) {
    def containerID = "0"
} else {
    def containerID = new File(instanceName+"_containerID").text.trim()
}

def containerUrl = hostname+":"+port
if (httpsEnabled != null){
    if ("true".equalsIgnoreCase(httpsEnabled)){
        containerUrl = "https://"+containerUrl
    }else{
        containerUrl = "http://"+containerUrl
    }
}else{
    if (httpEnabled != null && "true".equalsIgnoreCase(httpEnabled)){
        containerUrl = "http://"+containerUrl
    }
}
println "containerUrl: " + containerUrl

variables.put("HOSTNAME", hostname)
variables.put("PORT", port)

// Determine Cloud Automation URL
def paSchedulerRestUrl = variables.get('PA_SCHEDULER_REST_URL') 
def pcaUrl = paSchedulerRestUrl.replaceAll("/rest\\z", "/cloud-automation-service")

try {
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
    endpoint.setUrl(containerUrl);
    // Set the endpoint parameters according to the Proxy settings
    if (proxyfied.toLowerCase()=="true"){
        proxyfiedURL = pcaUrl+"/services/"+instanceId+"/endpoints/"+endpointID+"/"
        endpoint.setProxyfied(true);
        endpoint.setProxyfiedUrl(proxyfiedURL)
    }else{
        endpoint.setProxyfied(false)
    }

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

    // Inform other platforms that service is running through Synchronization API
    channel = "Service_Instance_" + instanceId
    synchronizationapi.createChannelIfAbsent(channel, true)
    synchronizationapi.put(channel, "RUNNING", true)
    synchronizationapi.put(channel, "INSTANCE_NAME", instanceName)

    // Add token to the current node
    token = instanceName
    nodeUrl = variables.get("PA_NODE_URL")
    println("Current nodeUrl: " + nodeUrl)
    println("Adding token:    " + token)
    rmapi.connect()
    rmapi.addNodeToken(nodeUrl, token)

    // Log output
    println(variables.get("PA_JOB_NAME") + "_INSTANCE_ID: " + instanceId)
    println(variables.get("PA_JOB_NAME") + "_ENDPOINT: " + endpoint)
} catch (Exception e) {
    StackTraceUtils.printSanitizedStackTrace(e)
    throw e
}

println("END " + variables.get("PA_TASK_NAME"))