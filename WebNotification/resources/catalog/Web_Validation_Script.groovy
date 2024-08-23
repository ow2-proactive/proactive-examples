// Skip web validation (used for automatic workflow tests to skip manual validation)
if (variables.get("SKIP_WEB_VALIDATION") && variables.get("SKIP_WEB_VALIDATION").toBoolean()) {
    sleep(10000)
    println("Skipping web validation ...");
    return;
}

import org.ow2.proactive.notification.client.ApiClient
import org.ow2.proactive.notification.client.api.ValidationRestApi
import org.ow2.proactive.notification.client.model.ValidationRequest
import org.ow2.proactive.notification.client.ApiException
import org.apache.commons.lang3.StringUtils

Set<String> authorizedUsers = Optional.ofNullable(variables.get("AUTHORIZED_USERS"))
        .map { it.split("\\s*,\\s*")}
        .orElse(Collections.emptySet())

Set<String> authorizedGroups = Optional.ofNullable(variables.get("AUTHORIZED_GROUPS"))
        .map { it.split("\\s*,\\s*")}
        .orElse(Collections.emptySet())

boolean isJobSubmitterAuthorized;
try {
    isJobSubmitterAuthorized = Boolean.parseBoolean(variables.get("IS_JOB_SUBMITTER_AUTHORIZED"));
} catch (Exception ignored) {
    // Either the variable is not present or not a valid boolean. Default to true
    isJobSubmitterAuthorized = true;
}

// Check that at least one user will see the Validation
if (!isJobSubmitterAuthorized && (authorizedGroups.isEmpty() || (authorizedGroups.size() == 1 && StringUtils.isBlank(authorizedGroups.toArray()[0]))) && (authorizedUsers.isEmpty() || (authorizedUsers.size() == 1 && StringUtils.isBlank(authorizedUsers.toArray()[0])))) {
    throw new IllegalArgumentException("At least the Job submitter, a user or a user group must be authorized on the Validation")
}

//Get notification-service URL
def notifUrl = variables.get('PA_NOTIFICATION_SERVICE_REST_URL')

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

// Get event severity or set default
def eventSeverity = variables.get("SEVERITY")
if (eventSeverity == null || eventSeverity.isEmpty()) {
    eventSeverity = ValidationRequest.EventSeverityEnum.WARNING
} else {
    eventSeverity = ValidationRequest.EventSeverityEnum.valueOf(eventSeverity)
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
        .authorizedUsers(authorizedUsers)
        .authorizedGroups(authorizedGroups)
        .isCreatedByAuthorizedToAction(isJobSubmitterAuthorized)
        .jobId(jobId)
        .message(validationMessage)
        .eventSeverity(eventSeverity)

try {
    validationRestApi.createValidation(sessionId, validationRequest)
    println("Validation request sent!")
} catch (ApiException e) {
    println("[WARNING] Something went wrong while creating the Web Validation. Please manually handle the paused Job")
    e.printStackTrace();
}
