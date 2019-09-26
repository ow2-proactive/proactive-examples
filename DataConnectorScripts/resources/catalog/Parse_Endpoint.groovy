def dbmsName = args[0]
def endpoint = new URL(results[0].toString())
def user = variables.get("USER")
def host = endpoint.getHost()
def port = endpoint.getPort()
def credentialsKey = dbmsName + "://" + user + "@" + host + ":" + port
variables.put("HOST", host)
variables.put("PORT", port)
variables.put("CREDENTIALS_KEY", credentialsKey)
// Wait for database sever to be up and fully running.
sleep(3000)