import org.ow2.proactive.pca.service.client.ApiClient
import org.ow2.proactive.pca.service.client.api.ServiceInstanceRestApi

// Arguments must be <service_id> <var0_name_to_store> <var0_to_retrieve> <var1_name_to_store> <var1_to_retrieve> ...
if (args.length != 2) {
    println("[set_service_FINISHED_if_all_clean_OK] ERROR: Number of arguments must be == 2")
    throw new IllegalArgumentException("Number of arguments must be == 2")
}

def action = args[0]
def clean_status_file_name = args[1]

// Retrieve variables
def pca_url = variables.get('PA_CLOUD_AUTOMATION_REST_URL')
def instance_id = variables.get("PCA_INSTANCE_ID") as long

// Get schedulerapi access and acquire session id
schedulerapi.connect()
def session_id = schedulerapi.getSession()

// Connect to Cloud Automation API
def service_instance_rest_api = new ServiceInstanceRestApi(new ApiClient().setBasePath(pca_url))

// Get the status
def all_clean_status = new File(clean_status_file_name).text
def any_clean_failed = all_clean_status.split("\\|").any{ cmd_err -> cmd_err == "ko"  }
def current_status = any_clean_failed ? "ERROR" : action

// Update service instance data : (status, endpoint)
def service_instance_data = service_instance_rest_api.getServiceInstance(session_id, instance_id)
service_instance_data.setInstanceStatus(current_status)
service_instance_rest_api.updateServiceInstance(session_id, instance_id, service_instance_data)

if(action.equals("FINISHED")){
    // Inform other jobs that the service is finished and deleted.
    def channel = "Service_Instance_" + instance_id
    synchronizationapi.put(channel, "FINISH_DONE", true)
}

// Print warning or error messages and force job to exit with error if there are any.
if (any_clean_failed){
    println "[set_service_FINISHED_if_all_clean_OK] ERROR: clean failed"
    throw new IllegalStateException("Clean failed")
}