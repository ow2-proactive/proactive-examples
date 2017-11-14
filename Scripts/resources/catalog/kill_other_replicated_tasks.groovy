schedulerapi.connect()

def nb_runs = variables.get("nb_runs") as Integer
def job_id = variables.get("PA_JOB_ID")
def task_id = variables.get("PA_TASK_REPLICATION") as Integer
def task_name = variables.get("PA_TASK_NAME")
		
// Now kill the others (deadlock safe??)
def task_basis_name = task_name.split("\\*")[0]
(0..(nb_runs-1)).each { tid ->
  
	// Do not kill myself
	if (tid != task_id)	
	{  
		def task_name_to_kill = task_basis_name + "*" + tid
		if (tid == 0)
			task_name_to_kill = task_basis_name
		
		//println task_name + " will kill " + task_name_to_kill
		schedulerapi.killTask(job_id, task_name_to_kill)
	}
}