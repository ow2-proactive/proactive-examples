def instanceId = variables.get("PCA_INSTANCE_ID") as long
def instanceName = variables.get("INSTANCE_NAME")
def endpointID = variables.get("ENDPOINT_ID")
    
def paSchedulerRestUrl = variables.get('PA_SCHEDULER_REST_URL')
def pcaUrl = paSchedulerRestUrl.replaceAll("/rest\\z", "/cloud-automation-service")    

proxified = variables.get("PROXYFIED").toLowerCase()
https_enabled = variables.get("HTTPS_ENABLED").toLowerCase()
variables.put("PROXYFIED", proxified)
variables.put("HTTPS_ENABLED", https_enabled)

if (proxified == "true") {
    proxyfiedURL = pcaUrl+"/services/"+instanceId+"/endpoints/"+endpointID+"/"
	wsURL = proxyfiedURL.replace("https://", "wss://")
	wsURL = wsURL.replace("http://", "ws://")
	println "Proxyfied URL :" + proxyfiedURL
	println "WebSocket URL :" + wsURL
	variables.put("PROXYFIED_URL", proxyfiedURL)
	variables.put("WS_PROXYFIED_URL", wsURL)
}