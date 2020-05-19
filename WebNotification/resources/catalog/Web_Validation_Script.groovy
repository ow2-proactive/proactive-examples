// Skip web validation (used for automatic workflow tests to skip manual validation)
if (variables.get("SKIP_WEB_VALIDATION") && variables.get("SKIP_WEB_VALIDATION").toBoolean()) {
    println("Skipping web validation ...");
    return;
}

import org.ow2.proactive.notification.client.ApiClient
import org.ow2.proactive.notification.client.api.ValidationRestApi
import org.ow2.proactive.notification.client.model.ValidationRequest
import org.ow2.proactive.notification.client.model.Validation
import org.ow2.proactive.notification.client.ApiException

//Get notification-service URL
def paSchedulerRestUrl = variables.get('PA_SCHEDULER_REST_URL')
def notifUrl = paSchedulerRestUrl.replaceAll("/rest\\z", "/notification-service")

// Instantiate ValidationRestApi instance
def apiClient = new ApiClient()
apiClient.setBasePath(notifUrl)
def validationRestApi = new ValidationRestApi(apiClient)

//Get job id
def jobId = new Long(variables.get("PA_JOB_ID"))
println(jobId)
println(variables.get("PA_JOB_ID"))

//Get validation message
def validationMessage = variables.get("MESSAGE")
if (validationMessage == null || validationMessage.isEmpty()) {
    validationMessage = "Validation request custom message."
}

//Get session id
schedulerapi.connect()
def sessionId = schedulerapi.getSession()

// Pause job
schedulerapi.pauseJob(variables.get("PA_JOB_ID"))

//Create validation request
def validationRequest = new ValidationRequest()
        .bucketName(genericInformation.get("bucketName"))
        .workflowName(variables.get("PA_JOB_NAME"))
        .jobId(jobId)
        .message(validationMessage)

try {
    Validation result = validationRestApi.createValidationUsingPOST(sessionId, validationRequest)
    println("Validation request sent!")
} catch (ApiException e) {
    println("[WARNING] Something went wrong while creating the Web Validation")
    e.printStackTrace();
}
