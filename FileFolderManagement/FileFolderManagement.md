File and Folder Management
========================

## Workflow overview

The workflows in this package are presenting two different ways to access files and folders. 

* In the LocalFolderCheck example, the folder is accessed via its absolute path on the computer it is executed in. This require the file or folder to exist.
* In the RemoteFileCheck example, the file is within the shared directory managed by ProActive. The feature dataspace is used in that specific case.

Overall, the few examples in this workflows are presenting:

* A trigger based on number of files within a folder
* A trigger based on overall folder size
* A trigger based on a date which check if a file has been edited after a specific date
* A trigger based on the number of lines within a file
