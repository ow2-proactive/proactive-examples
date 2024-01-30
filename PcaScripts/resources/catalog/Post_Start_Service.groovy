/*********************************************************************************
* THIS POSTSCRIPT PROPAGATES USEFUL INFORMATION SUCH AS:                         *
* 1) SERVICE ENDPOINT (PROTOCOL://HOSTNAME:PORT)                                 *
* 2) CREDENTIALS (IF THERE ARE ANY) BY ADDING THEM TO 3RD PARTY CREDENTIALS      *
*********************************************************************************/

import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.pca.service.client.api.IpUtilsRestApi
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
def httpEnabled = variables.get("HTTP_ENABLED") // e.g. Visdom, Tensorboard
def httpsEnabled = variables.get("HTTPS_ENABLED") // e.g. MaaS, JupyterLab
def engine = variables.get("ENGINE") // docker, singularity

try {

    // Determine Cloud Automation URL
    def pcaUrl = variables.get('PA_CLOUD_AUTOMATION_REST_URL')

    // Get schedulerapi access and acquire session id
    schedulerapi.connect()
    def sessionId = schedulerapi.getSession()

    // Connect to Cloud Automation API
    def apiClient = new ApiClient()
    apiClient.setBasePath(pcaUrl)
    def serviceInstanceRestApi = new ServiceInstanceRestApi(apiClient)

    def ipUtilsRestApi = new IpUtilsRestApi(apiClient)

    // Handle service parameters
    def hostname = ipUtilsRestApi.getMyRemotePublicAddr(sessionId)
    if (hostname.equals("127.0.0.1")) {
        hostname = variables.get("PA_NODE_HOST")
    }
    def port = new File(instanceName+"_port").text.trim()

    def containerID="";
    if ("docker".equalsIgnoreCase(engine)) {
        containerID = new File(instanceName+"_containerID").text.trim()
        if ("".equals(containerID)) {
            println("Docker container didn't started, terminating execution.")
            return;
        }
    }

    def containerUrl = hostname + ":" + port
    if (httpsEnabled != null){
        if ("true".equalsIgnoreCase(httpsEnabled)){
            containerUrl = "https://"+containerUrl
        }else{
            containerUrl = "http://"+containerUrl
        }
    }else{
        if (httpEnabled != null && "true".equalsIgnoreCase(httpEnabled)){
            containerUrl = "http://" + containerUrl
        } else if(binding.variables["args"] && args.length > 0 && !args[0].trim().isEmpty()){
            containerUrl = args[0] + "://" + containerUrl
        } else {
            containerUrl = null;
        }
    }

    println "containerUrl: " + containerUrl

    variables.put("HOSTNAME", hostname)
    variables.put("PORT", port)

    // Implement service model

    // Container
    def Container container = new Container()
    container.setId(containerID)
    container.setName(instanceName)

    // Endpoint
    def Endpoint endpoint;
    if(containerUrl != null){
        endpoint = new Endpoint();
        endpoint.setId(endpointID);
        endpoint.setUrl(containerUrl);
        // Set the endpoint parameters according to the Proxy settings
        if (proxyfied != null){
            if (proxyfied.toLowerCase()=="true"){
                proxyfiedURL = variables.get('PROXYFIED_URL')
                endpoint.setProxyfied(true);
                endpoint.setProxyfiedUrl(proxyfiedURL)
            }else{
                endpoint.setProxyfied(false)
            }
        }
    }

    // Node
    def Node node = new Node();
    node.setName(variables.get("PA_NODE_NAME"))
    node.setHost(variables.get("PA_NODE_HOST"))
    node.setNodeSourceName(variables.get("PA_NODE_SOURCE"))
    node.setUrl(variables.get("PA_NODE_URL"))

    // Deployment
    def Deployment deployment;
    if(endpoint != null){
        deployment = new Deployment()
        deployment.setNode(node)
        deployment.setContainer(container)
        deployment.setEndpoint(endpoint)
    }

    // Update service instance model (add Deployment, Groups)
    def serviceInstanceData = serviceInstanceRestApi.getServiceInstance(sessionId, instanceId)
    serviceInstanceData.setInstanceStatus("RUNNING")
    if(deployment != null){
        serviceInstanceData = serviceInstanceData.addDeploymentsItem(deployment)
    }
    if (proxyfied != null && proxyfied.toLowerCase()=="true"){
        serviceInstanceData = serviceInstanceData.addGroupsItem("scheduleradmins")
        serviceInstanceData = serviceInstanceData.addGroupsItem("rmcoreadmins")
    }
    serviceInstanceData = serviceInstanceRestApi.updateServiceInstance(sessionId, instanceId, serviceInstanceData)
    println(serviceInstanceData)

    schedulerapi.registerService(variables.get("PA_JOB_ID"), instanceId as int, true)

    // Inform other platforms that service is running through Synchronization API
    channel = "Service_Instance_" + instanceId
    synchronizationapi.createChannelIfAbsent(channel, true)
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
    println(variables.get("PA_JOB_NAME") + "_ENDPOINT: " + endpoint !=null ? endpoint: "")
} catch (Exception e) {
    StackTraceUtils.printSanitizedStackTrace(e)
    throw e
}

println("END " + variables.get("PA_TASK_NAME"))