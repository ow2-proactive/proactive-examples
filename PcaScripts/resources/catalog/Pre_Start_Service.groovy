def instanceId = variables.get("PCA_INSTANCE_ID") as long
def instanceName = variables.get("INSTANCE_NAME")
def endpointID = variables.get("ENDPOINT_ID")
def proxified = variables.get("PROXYFIED")

def pcaPublicUrl = variables.get('PA_CLOUD_AUTOMATION_REST_PUBLIC_URL')

if ("true".equalsIgnoreCase(proxified)) {
    proxyfiedURL = pcaPublicUrl+"/services/"+instanceId+"/endpoints/"+endpointID+"/"
    wsURL = proxyfiedURL.replace("https://", "wss://")
    wsURL = wsURL.replace("http://", "ws://")
    println "Proxyfied URL :" + proxyfiedURL
    println "WebSocket URL :" + wsURL
    variables.put("PROXYFIED_URL", proxyfiedURL)
    variables.put("WS_PROXYFIED_URL", wsURL)
}