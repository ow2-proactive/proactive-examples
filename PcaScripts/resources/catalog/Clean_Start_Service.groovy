/*********************************************************************************
 * THIS CLEANCRIPT MAKE SURE THAT THE PCA INSTANCE STATUS IS IN ERROR IN CASE THE *
 * SERVICE HAS NOT STARTED PROPERLY                                               *
 *********************************************************************************/

import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi
import org.ow2.proactive.pca.service.client.model.ServiceInstanceData

// Acquire variables
def instanceId = variables.get("PCA_INSTANCE_ID") as long

// Determine Cloud Automation URL
def pcaUrl = variables.get('PA_CLOUD_AUTOMATION_REST_URL')

// Get schedulerapi access and acquire session id
schedulerapi.connect()
def sessionId = schedulerapi.getSession()

// Connect to Cloud Automation API
def apiClient = new ApiClient()
apiClient.setBasePath(pcaUrl)
def serviceInstanceRestApi = new ServiceInstanceRestApi(apiClient)


// Update service instance model (add Deployment, Groups)
def serviceInstanceData = serviceInstanceRestApi.getServiceInstance(sessionId, instanceId)

// If service instance is NOT RUNNING then it should be in ERROR
def currentStatus = serviceInstanceData.getInstanceStatus()
if (!currentStatus.equals("RUNNING")) {
    serviceInstanceData.setInstanceStatus("ERROR")
    serviceInstanceData = serviceInstanceRestApi.updateServiceInstance(sessionId, instanceId, serviceInstanceData)
    println(serviceInstanceData)
}