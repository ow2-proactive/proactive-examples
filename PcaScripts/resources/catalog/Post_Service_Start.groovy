def instanceName = variables.get("INSTANCE_NAME")
def instanceId = variables.get("INSTANCE_ID_" + instanceName)
def enableActions = args[0]

// Get schedulerapi access and acquire session id
schedulerapi.connect()
schedulerapi.registerService(variables.get("PA_JOB_ID").toString(), (int) instanceId, enableActions.toBoolean())
println("Service instance " + instanceId + "attached to the parent job " + variables.get("PA_JOB_ID"))