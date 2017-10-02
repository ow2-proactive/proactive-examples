CNES, PEPS Project
==================

## Context

ActiveEon is working with the CNES on the [PEPS project](https://peps.cnes.fr/rocket/#/home). For more info, please do not hesitate to contact us.

## Workflow overview

The workflows in this package is presenting one way to interact with a third party solution. This workflow takes advantage of a lot of features included in ProActive and will require some advanced understanding to fully grasp the full extent of the solution.

Some key features of ProActive are used in this example:

* [Error management](https://www.activeeon.com/public_content/documentation/7.29.0/user/ProActiveUserGuide.html#_handling_failures) with granular control
* Shared directory with the [dataspace](https://www.activeeon.com/public_content/documentation/7.29.0/user/ProActiveUserGuide.html#_scheduler_and_dataspace_apis)
* [Variable propagation](https://www.activeeon.com/public_content/documentation/7.29.0/user/ProActiveUserGuide.html#_variable_based_propagation) 
* Distribution with a user friendly [replication system](https://www.activeeon.com/public_content/documentation/7.29.0/user/ProActiveUserGuide.html#_replicate)
* [Docker](https://www.activeeon.com/public_content/documentation/7.29.0/user/ProActiveUserGuide.html#_docker_fork_execution_environment) support

**Note**: The PEPS platform is a public platform that often require maintenance, their service is not always available and consequently this workflow is impacted.

## In details

### Cnes

Task - check_if_docker_is_installed: Kill the overall job if docker is not installed. This has a different error management rule than the rest of the workflow. It is also possible to use a [selection script](https://www.activeeon.com/public_content/documentation/7.29.0/user/ProActiveUserGuide.html#_selection) to avoid this step.

Task - split: Create the arrays that will be used by the replication controller.

Task - submit_request_to_ext_service: Send request to external service to get xml.

Task - get_status_location: Parse xml to identify url for status.

Task - wait: Wait for third party solution to complete its own process.

Task - extract_tif_url: Parse the xml resulting from the process to extract image url.

Task - dl_and_conversion: Download image and convert it using a library within a container environment.

Task - end_block: End of the replication control and enable [image viewing](https://www.activeeon.com/public_content/documentation/7.29.0/user/ProActiveUserGuide.html#_retrieve_results_from_the_portal) within the browser.

Task - merge: Merge all the image names to transfer to the next task.

Task - jpg_merge: Merge the images with a library within a container.

Task - preview_merged_jpg: Enable preview of final merged image
