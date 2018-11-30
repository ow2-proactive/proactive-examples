/*********************************************************************************
* THIS PRESCRIPT INFORMS PLATFORM THAT PCA SERVICE ACTION IS TRIGGERED                   *
*********************************************************************************/

println("BEGIN " + variables.get("PA_TASK_NAME"))

action = args[0]

// Acquire service instance id and instance name from synchro channel
def instanceId = variables.get("PCA_INSTANCE_ID") as long
def channel = "Service_Instance_" + instanceId
def instanceName = synchronizationapi.get(channel, "INSTANCE_NAME")
variables.put("INSTANCE_NAME", instanceName)

// Inform other platforms that service action is being triggred through Synchronization API
synchronizationapi.put(channel, action, true)