import org.ow2.proactive.notification.client.ApiClient
import org.ow2.proactive.notification.client.api.EventRestApi
import org.ow2.proactive.notification.client.model.EventRequest
import org.ow2.proactive.notification.client.model.Event
import org.ow2.proactive.notification.client.ApiException

// Skip web notification
if (variables.get("ACTIVATE_NOTIFICATIONS") && !variables.get("ACTIVATE_NOTIFICATIONS").toBoolean()) {
    println("Skipping web notification ...");
    return;
}

//Get notification-service URL
def notifUrl = variables.get('PA_NOTIFICATION_SERVICE_REST_URL')

// Instantiate EventRestApi instance
def apiClient = new ApiClient()
apiClient.setBasePath(notifUrl)
def eventRestApi = new EventRestApi(apiClient)

//Get job id
def jobId = new Long(variables.get("PA_JOB_ID"))

//Get event message or set default
def eventMessage = variables.get("MESSAGE")
if (eventMessage == null || eventMessage.isEmpty()) {
    eventMessage = "You have a notification."
}

// Get event severity or set default
def eventSeverity = variables.get("SEVERITY")
if (eventSeverity == null || eventSeverity.isEmpty()) {
    eventSeverity = EventRequest.EventSeverityEnum.INFO
} else {
    eventSeverity = EventRequest.EventSeverityEnum.valueOf(eventSeverity)
}

//Get session id
schedulerapi.connect()
def sessionId = schedulerapi.getSession()


//Create event
def eventRequest = new EventRequest()
        .bucketName(genericInformation.get("bucketName"))
        .workflowName(variables.get("PA_JOB_NAME"))
        .eventType(EventRequest.EventTypeEnum.CUSTOM)
        .eventSeverity(eventSeverity)
        .jobId(jobId)
        .message(eventMessage);

try {
    result = eventRestApi.createEvent(sessionId, eventRequest).toString()
    println(result)
} catch (ApiException e) {
    System.err.println("Exception when calling EventRestApi#createEvent")
    e.printStackTrace();
}