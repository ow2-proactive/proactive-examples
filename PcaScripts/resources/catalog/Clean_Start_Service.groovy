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
def paSchedulerRestUrl = variables.get('PA_SCHEDULER_REST_URL')
def pcaUrl = paSchedulerRestUrl.replaceAll("/rest\\z", "/cloud-automation-service")

// Connect to Cloud Automation API
def apiClient = new ApiClient()
apiClient.setBasePath(pcaUrl)
def serviceInstanceRestApi = new ServiceInstanceRestApi(apiClient)


// Update service instance model (add Deployment, Groups)
def serviceInstanceData = serviceInstanceRestApi.getServiceInstanceUsingGET(instanceId)

// If service instance is NOT RUNNING then it should be in ERROR
def currentStatus = serviceInstanceData.getInstanceStatus()
if (!currentStatus.equals("RUNNING")) {
    serviceInstanceData.setInstanceStatus("ERROR")
    serviceInstanceData = serviceInstanceRestApi.updateServiceInstanceUsingPUT(instanceId, serviceInstanceData)
    println(serviceInstanceData)
}