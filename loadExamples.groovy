import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import java.util.zip.ZipFile

println "Automatic deployment of proactive examples ..."

def unzipFile(src, dest) {
    def zipFile = new ZipFile(src)
    zipFile.entries().each { it ->
        def path = Paths.get(dest + "/" + it.name)
        if(it.directory){
            Files.createDirectories(path)
        }
        else {
            def parentDir = path.getParent()
            if (!Files.exists(parentDir)) {
                Files.createDirectories(parentDir)
            }
            Files.copy(zipFile.getInputStream(it), path)
        }
    }
}

// Retrieve variables
def examples_zip_path = "/home/michael/TEST/proactive-examples.zip"
def example_dir_path = "/home/michael/TEST/proactive-examples"
def global_space_path = this.binding.variables.get("pa.scheduler.dataspace.defaultglobal.localpath")
def scheduler_rest_url = this.binding.variables.get("pa.scheduler.rest.url")
def workflow_catalog_url = scheduler_rest_url.substring(0,scheduler_rest_url.length()-4) + "workflow-catalog"
def workflow_templates_dir_path = "/home/michael/scheduling/config/workflows/templates"
def bucket_owner = "workflow-catalog"
	
println "examples_zip_path " + examples_zip_path
println "example_dir_path " + example_dir_path
println "global_space_path " + global_space_path
println "workflow_catalog_url " + workflow_catalog_url
println "workflow_templates_dir_path " + workflow_templates_dir_path
println "bucket_owner " + bucket_owner

// If the unzipped dir already exists, lets remove it
def example_dir = new File(example_dir_path)
if (example_dir.exists())
{
	example_dir.deleteDir()
	println "Existing " + example_dir_path + " deleted!"
}

// Unzip the examples
def examples_zip = new File(examples_zip_path)
unzipFile(examples_zip, example_dir_path)
println examples_zip_path + " extracted!"

def target_dir_path = ""
def bucket = ""

// Start by finding the next templates dir index
def templates_dirs_list = []
new File(workflow_templates_dir_path).eachDir { dir ->
	templates_dirs_list << dir.getName().toInteger()
}
def template_dir_name = (templates_dirs_list.sort().last() + 1) + ""
println "Next template dir name " + template_dir_name


// For each example directory
example_dir.eachDir() { dir ->  

	def metadata_file = new File(dir.absolutePath, "METADATA.json")
	if (metadata_file.exists())
	{
		println "Parsing " + metadata_file.absolutePath
		
		// From json to map
		def slurper = new groovy.json.JsonSlurper()
		def metadata_file_map = (Map) slurper.parseText(metadata_file.text)
		
		def catalog_map = metadata_file_map.get("catalog")
		
		
		// DATASPACE SECTION /////////////////////////////
		
		
		if ((dataspace_map = catalog_map.get("dataspace")) != null)
		{
			// Retrieve the targeted directory path
			def target = dataspace_map.get("target")
			if(target == "global")
			target_dir_path = global_space_path
			
			// Copy all files into the targeted directory
			dataspace_map.get("files").each { file_relative_path ->
				def file_src = new File(dir.absolutePath, file_relative_path)
				def file_src_path = file_src.absolutePath
				def file_name = file_src.getName()
				def file_dest = new File(target_dir_path, file_name)	
				def file_dest_path = file_dest.absolutePath
				Files.copy(Paths.get(file_src_path), Paths.get(file_dest_path), StandardCopyOption.REPLACE_EXISTING)
			}
		}
		
		
		// BUCKET SECTION /////////////////////////////
		
		
		bucket = catalog_map.get("bucket")
		
		// Does the bucket already exist? -------------
		def list_buckets_cmd = [ "bash", "-c", "curl -X GET --header 'Accept: application/json' '" + workflow_catalog_url + "/buckets?owner=" + bucket_owner + "'"]
		def response = new StringBuilder()
		println "Executing " + list_buckets_cmd
		
		list_buckets_cmd.execute().waitForProcessOutput(response, System.err)

		def bucket_id = -1
		// Test if the buckets list is empty
		if (slurper.parseText(response.toString()).get("_embedded") != null)
		{
			slurper.parseText(response.toString()).get("_embedded").get("bucketMetadataList").each { bucketMetada ->
				if (bucketMetada.get("name") == bucket)
					bucket_id = bucketMetada.get("id")
					// Cannot break in a closure			
			}
		}
		println "bucket_id retrieved: " + bucket_id
		
		
		// Create a bucket if needed -------------
		if (bucket_id == -1)
		{
			def create_bucket_cmd = [ "bash", "-c", "curl -X POST --header 'Content-Type: application/json' --header 'Accept: application/json' '" + workflow_catalog_url + "/buckets?name=" + bucket + "&owner=" + bucket_owner + "'"]
			response = new StringBuilder()
			println "Executing " + create_bucket_cmd
			create_bucket_cmd.execute().waitForProcessOutput(response, System.err)
			bucket_id = slurper.parseText(response).get("id")
			println "bucket_id created: " + bucket_id
		}
		
		
		// OBJECTS SECTION /////////////////////////////

		
		catalog_map.get("objects").each { object ->
			def metadata_map = object.get("metadata")
			
			
			// WORKFLOWS SECTION /////////////////////////////
			
			
			if (metadata_map.get("kind") == "workflow")
			{
				def workflow_relative_path = object.get("file")
				def workflow_file_name = new File(workflow_relative_path).getName()
				def workflow_absolute_path = new File(dir.absolutePath, workflow_relative_path).absolutePath

				// Push the workflow to the bucket
				def push_wkw_cmd = [ "bash", "-c", "curl -X POST --header 'Content-Type: multipart/form-data' --header 'Accept: application/json' '" + workflow_catalog_url + "/buckets/" + bucket_id + "/workflows' -F 'file=@" + workflow_absolute_path + "'"]
				println "Executing " + push_wkw_cmd
				push_wkw_cmd.execute().waitForProcessOutput(System.out, System.err)
				
				if (object.get("expose_to_studio") == "yes")
				{
					// Create a new template dir in the targeted directory and copy the wkw into it
					def template_dir = new File(workflow_templates_dir_path, template_dir_name)
					template_dir.mkdir()
					def file_dest = new File(template_dir, workflow_file_name)	
					def file_dest_path = file_dest.absolutePath
					Files.copy(Paths.get(workflow_absolute_path), Paths.get(file_dest_path), StandardCopyOption.REPLACE_EXISTING)
					
					template_dir_name = (template_dir_name.toInteger() + 1) + ""
				}
			}
				
		}
			
	}
}

println "... proactive examples deployed!"
	
