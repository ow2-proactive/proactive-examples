/*
#If you want to add more functionalities like a Proxy use
# Please refer to Commons Virtual File System doc for more info.
# https://commons.apache.org/proper/commons-vfs/index.html
*/

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.regex.Pattern;
import org.apache.commons.vfs2.*;
import org.apache.commons.vfs2.auth.*;
import org.apache.commons.vfs2.impl.*;
import org.apache.commons.vfs2.provider.local.*;
import org.apache.commons.vfs2.provider.ftp.FtpFileSystemConfigBuilder;
import org.objectweb.proactive.extensions.dataspaces.vfs.selector.*

URI_SCHEME = args[0]
schedulerapi.connect()

filesToProcess = new ArrayList();
channelId = variables.get("CHANNEL_TRIGGERED_ID")

///Set connection parameters and retrieve the SFTP/FTP password
URL_KEY = URI_SCHEME + "://<username>@<host>";
host = variables.get("HOST")
username = variables.get("USERNAME")
port = variables.get("PORT")

if (signalapi.isReceived("Terminate_Monitoring")) {    
    result = true;
    return;
}

password = checkParametersAndReturnPassword()

//Initialize the connection manger to the remote SFTP/FTP server.
optsRemote = new FileSystemOptions()
fsManager = null
initializeAuthentication()

//Initialize file pattern, local and remote bases
remoteDir = variables.get("REMOTE_BASE")
filePattern = variables.get("FILE_PATTERN")
if (filePattern.isEmpty()) {
    throw new IllegalArgumentException("FILE_PATTERN variable is not provided by the user. Empty value is not allowed.")
}

// used for cleanup in release()
src = null

//Export file(s) to the SFTP/FTP server
importFiles()
release()
schedulerapi.disconnect()

/**
 * Retrieves files that match the specified File pattern from the SFTP/FTP server
 * and import them to the local directory (data space).
 */
void importFiles() {
    try {
        if (port == null || port.isEmpty()) {
            startUrl = URI_SCHEME + "://" + host + "/" + remoteDir
        } else {
            startUrl = URI_SCHEME + "://" + host + ":" + port + "/" + remoteDir
        }
        remoteFile = fsManager.resolveFile(startUrl, optsRemote)
        // Set src for cleanup in release()
        src = remoteFile
        remoteBasePath = remoteFile.getName()
        children = this.remoteFile.findFiles(new org.objectweb.proactive.extensions.dataspaces.vfs.selector.FileSelector(filePattern))
        children.each { f ->
            String relativePath = File.separator + remoteBasePath.getRelativeName(f.getName());
            if (f.getType() == FileType.FILE) {
                key = relativePath                
                value = true // we no need value, so it is always true
                previousValue = synchronizationapi.putIfAbsent(channelId, key, value)
                if (previousValue == null) {

                    // we found a new file which was apparently added

                    // Submit a workflow referenced in the ProActive Catalog (or accessible by url)
                    // Please add the bucket and workflow names that you want to execute as Task Variables 'CALLED_WORKFLOW'

                    // get the workflow information from Task Variables, e.g. 'basic-examples/Native_Task'
                    calledWorkflow = variables.get("CALLED_WORKFLOW")

                    // variables submitted to the workflow: Empty in the example, modify this map to give parameters to the submitted workflow
                    variableName = variables.get("FOUND_FILE_VARIABLE_NAME")
                    workflow_variables = [(variableName) : key]
                    def generic_infos_map = ["PARENT_JOB_ID" : variables.get("PA_JOB_ID")]

                    println "Submitting workflow " + calledWorkflow
                    // submitting the job

                    jobid = schedulerapi.submitFromCatalog(variables.get("PA_CATALOG_REST_URL"), calledWorkflow, workflow_variables, generic_infos_map)

                    println "Job submitted with job id " + jobid

                }    			
            }
        }
    } catch (FileSystemException ex) {
        throw new RuntimeException(ex);
    }
}


/**
 * Release system resources, close connections to the local and remote filesystems.
 */
void release() {
    FileSystem fs = null;    
    if (src != null) {
        src.close()
        fs = src.getFileSystem() // This works even if the src is closed.
        fsManager.closeFileSystem(fs)
    }
}

/**
 * Checks whether the provided host, username and port values are empty or not, then
 * returns the SFTP/FTP password using the third party credentials mechanism
 */
def checkParametersAndReturnPassword() {
    if (host.isEmpty()) {
        throw new IllegalArgumentException("HOST variable is not provided by the user. Empty value is not allowed.")
    }
    if (username.isEmpty()) {
        throw new IllegalArgumentException("USERNAME variable is not provided by the user. Empty value is not allowed.")
    }
    def urlKey = URI_SCHEME + "://" + username + "@" + host;
    def password = variables.get(urlKey)
    if (password == null || password.isEmpty()) {
        throw new IllegalArgumentException("Please add your " + URI_SCHEME + " password to 3rd-party credentials under the key :\"" +
                                           URL_KEY + "\"");
    }
    return password
}

/**
 * Initializes the connection to the remote SFTP/FTP server
 * Returns the FileSystemManager instance that manages this connection.
 */
void initializeAuthentication() {
    try {
        fsManager = VFS.getManager();
    } catch (FileSystemException ex) {
        throw new RuntimeException("Failed to get fsManager from VFS", ex);
    }
    def auth = new StaticUserAuthenticator(null, username, password)
    try {
        DefaultFileSystemConfigBuilder.getInstance().setUserAuthenticator(optsRemote, auth);
        FtpFileSystemConfigBuilder.getInstance().setPassiveMode(optsRemote, true);
    } catch (FileSystemException ex) {
        throw new RuntimeException("Failed to set user authenticator", ex);
    }
}