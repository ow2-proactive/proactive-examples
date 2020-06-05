def dbmsName = args[0]
def endpoint = new URL(results[0].toString())
def user = variables.get("USER")
def host = endpoint.getHost()
def port = endpoint.getPort()
def credentialsKey = dbmsName + "://" + user + "@" + host + ":" + port
variables.put("HOST", host)
variables.put("PORT", port)
variables.put("CREDENTIALS_KEY", credentialsKey)

// This value is based on an average estimation of how long it takes handled databases to be up
// Increase this value if this task fails at first attempt but succeeds at the second.
def SLEEP_TIME = 11000

// Wait for database sever to be up and fully running.
sleep(SLEEP_TIME)