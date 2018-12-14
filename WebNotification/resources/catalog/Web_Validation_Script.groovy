// Skip web validation (used for automatic workflow tests to skip manual validation)
if (variables.get("SKIP_WEB_VALIDATION") && variables.get("SKIP_WEB_VALIDATION").toBoolean()) {
    println("Skipping web validation ...");
    return;
}

// Please fill variables
def notification_message = args[0]

// Don't change code below unless you know what you are doing
def jobid = variables.get("PA_JOB_ID")
def userName = variables.get("PA_USER")

// pause job
schedulerapi.connect()
schedulerapi.pauseJob(jobid)

// send web validation
def notificationContent = '{"description": "' + notification_message + '", "jobId": "' + jobid + '" , "validation": "true", "userName":  "' + userName + '"}'
def notificationServiceURL = variables.get("PA_SCHEDULER_REST_URL").replace("/rest", "") + '/notification-service/notifications'
def post = new URL(notificationServiceURL).openConnection();
post.setRequestMethod("POST")
post.setDoOutput(true)
post.setRequestProperty("Content-Type", "application/json")
post.getOutputStream().write(notificationContent.getBytes("UTF-8"));
if (post.getResponseCode().equals(200)) {
    println("Web Validation sent!");
} else {
    println("[WARNING] Something went wrong while creating the Web Validation")
}