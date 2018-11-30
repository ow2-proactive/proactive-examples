/*********************************************************************************
* THIS PRESCRIPT INFORMS PLATFORM THAT PCA SERVICE IS BEING RESUMED              *
*********************************************************************************/

println("BEGIN " + variables.get("PA_TASK_NAME"))

// Acquire service instance id and instance name from synchro channel
def instanceId = variables.get("PCA_INSTANCE_ID") as long
def channel = "Service_Instance_" + instanceId
def instanceName = synchronizationapi.get(channel, "INSTANCE_NAME")
variables.put("INSTANCE_NAME", instanceName)

// Inform other platforms that service is running through Synchronization API
synchronizationapi.put(channel, "RUNNING", true)
synchronizationapi.put(channel, "RESUMED", true)
synchronizationapi.put(channel, "PAUSE_LAUNCHED", false)