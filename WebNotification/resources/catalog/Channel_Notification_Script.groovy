import org.ow2.proactive.notification.client.ApiClient
import org.ow2.proactive.notification.client.api.EventRestApi
import org.ow2.proactive.notification.client.model.EventRequest
import org.ow2.proactive.notification.client.model.Event
import org.ow2.proactive.notification.client.ApiException

//Get notification-service URL
def paSchedulerRestUrl = variables.get('PA_SCHEDULER_REST_URL')
def notifUrl = paSchedulerRestUrl.replaceAll("/rest\\z", "/notification-service")
// Instantiate Notification RestApi instance
def apiClient = new ApiClient()
apiClient.setBasePath(notifUrl)
def eventRestApi = new EventRestApi(apiClient)

//Get Job variables
def jobId = new Long(variables.get("PA_JOB_ID"))
def eventMessage = variables.get("MESSAGE")
def eventSeverity = variables.get("SEVERITY")
def channelsToNotify = variables.get("CHANNELS")

// Set notification message
eventMessage = (eventMessage == null || eventMessage.isEmpty()) ? "You have a notification.": eventMessage;
// Set channels to notify
channelsToNotify = (channelsToNotify != null && !channelsToNotify.equals('all')) ? new HashSet(Arrays.asList(channelsToNotify.split(','))): null;
// Set notification severity
eventSeverity = (eventSeverity == null || eventSeverity.isEmpty()) ? EventRequest.EventSeverityEnum.INFO: EventRequest.EventSeverityEnum.valueOf(eventSeverity);

//Get session id
schedulerapi.connect()
def sessionId = schedulerapi.getSession()

//Create event
def eventRequest = new EventRequest()
        .bucketName(genericInformation.get("bucketName"))
        .workflowName(variables.get("PA_JOB_NAME"))
        .eventType(EventRequest.EventTypeEnum.CHANNEL)
        .eventSeverity(eventSeverity)
        .channelsToNotify(channelsToNotify)
        .jobId(jobId)
        .message(eventMessage);

try {
    result = eventRestApi.createEvent(sessionId, eventRequest).toString()
    println(result)
} catch (ApiException e) {
    System.err.println("Exception when calling EventRestApi#createEvent")
    e.printStackTrace();
}