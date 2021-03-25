def endpoint = new URI(results[0].toString())
def host = endpoint.getHost()
def port = endpoint.getPort()
variables.put("HOST", host)
variables.put("PORT", port)

// This value is based on an average estimation of how long it takes handled databases to be up
// Increase this value if this task fails at first attempt but succeeds at the second.
def SLEEP_TIME = 50000

// Wait for database sever to be up and fully running.
sleep(SLEEP_TIME)
