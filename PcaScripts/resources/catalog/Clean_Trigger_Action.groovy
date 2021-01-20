/*********************************************************************************
*   THIS CLEANSCRIPT MAKES SURE THAT THE RM TOKEN IS CLEANED IF THE MAIN 		 *
*   WORKFLOW IS NOT RUNNING ANYMORE.										     *
*********************************************************************************/

import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.pca.service.client.model.ServiceInstanceData

// Acquire variables
def instanceId = variables.get("PCA_INSTANCE_ID") as long
def instanceName = variables.get("INSTANCE_NAME")

// Handle service parameters
def hostname = variables.get("PA_NODE_HOST")

// Determine Cloud Automation URL
def paSchedulerRestUrl = variables.get('PA_SCHEDULER_REST_URL')
def pcaUrl = paSchedulerRestUrl.replaceAll("/rest\\z", "/cloud-automation-service")

// Get schedulerapi access and acquire session id
schedulerapi.connect()
def sessionId = schedulerapi.getSession()

// Connect to Cloud Automation API
def apiClient = new ApiClient()
apiClient.setBasePath(pcaUrl)
def serviceInstanceRestApi = new ServiceInstanceRestApi(apiClient)

// Update service instance model (add Deployment, Groups)
def serviceInstanceData = serviceInstanceRestApi.getServiceInstanceUsingGET(sessionId, instanceId)

// Get the main
def submittedMainJobId = serviceInstanceData.getJobSubmissions().get(0).getJobId()
def mainJobState = schedulerapi.getJobState(submittedMainJobId.toString())

if (schedulerapi.getJobState(submittedMainJobId.toString()).isFinished()) {
    // Add token to the current node
    nodeUrl = variables.get("PA_NODE_URL")
    rmapi.connect()
    rmapi.removeNodeToken(nodeUrl, instanceName)
}