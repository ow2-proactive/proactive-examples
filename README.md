# ProActive Examples

This repository includes a sub directory per bucket, in which a metadata file centralizes object-related-information: bucket name, object name, object kind,..

The aim of this project is to centralize all proactive workflows and other related objects (scripts, images, etc). The workflows from the ProActive Examples project are pushed to Catalog storage inside proactive.

# How to build
Please run next command: ``gradlew clean zip`` or ``gradlew clean build``
This will generate the `proactive-examples.zip` file inside project's build folder.

# How to test locally
Copy the genarated proactive-examples.zip file to your `PA_SCHEDULER_HOME/samples` directory.
Start your proactive distribution. From this point everything should work ok.
During scheduling startup: the proactive-examples.zip archive will be extracted to `PA_SCHEDULER_HOME/samples/workflows/proactive-examples` folder. On the next step the special groovy script will automatically push the workflows from proactive-examples folder to Catalog storage.
If you need to retest the extracting and loading of proactive-examples, please remove the `samples/packages.loaded` file. Also to test the filling of catalog storage don't forget to clean database.

## The example of exact commands to test locally on linux:
```
1) PA_SCHEDULER_HOME is the path to your local Proactive distribution folder. You need to `cd` to this folder.
2) rm -f samples/packages.loaded
3) rm -fr data/*
4) you need to `cd` into your locally cloned proactive-examples project folder
5) ./gradlew clean zip
6) cp build/proactive-examples.zip PA_SCHEDULER_HOME/samples/
7) go back to PA_SCHEDULER_HOME and start proactive-server
8) ./PA_SCHEDULER_HOME/bin/proactive-server
```

# How to add a new package

1) Create a folder with the desired package name (e.g. **TextAnalysis**).

2) Add a `METADATA.json` file into the package (e.g. **TextAnalysis/METADATA.json**).

3) Insert the following JSON structure into the `METADATA.json` file:
```
{
	"metadata": {
		"slug": "textanalysis",
		"name": "Text Analysis",
		"short_description": "Text analysis with machine learning and deep learning on Docker",
		"author": "ActiveEon's Team",
		"tags": ["Samples", "Machine Learning", "Text Analysis", "Big Data", "Analytics", "Deep Learning"],
		"version": "1.0"
	},
	"catalog" : {
		"bucket" : "machine-learning",
		"objects" : [
			{
				"name" : "text_analysis",
				"metadata" : {
					"kind": "Workflow/standard",
					"commitMessage": "First commit",
					"contentType": "application/xml"
				},
				"file" : "resources/catalog/text_analysis.xml"
			}
		]
	}
}
```

3.1) Update the `metadata` fields:

* * metadata->**slug** - compact name of the package.
* * metadata->**name** - name of the package.
* * metadata->**short_description** - short description of the package.
* * metadata->**tags** - key works of the package.

3.2) Update the `catalog` fields:

* * catalog->**bucket** - Set the name of the bucket(s) for this package. A package can be installed in one or multiple buckets. Therefore, the bucket name can be a String value as in the example above (i.e `"Machine_Learning"`) or a JSON array value as in `["Bucket_1","Bucket_2","Bucket_X"]`.
* * catalog->**objects** - Add an object of each workflow of the package.

An example of a catalog object that represents a workflow:
```
{
				"name" : "text_analysis",
				"metadata" : {
					"kind": "Workflow/standard",
					"commitMessage": "First commit",
					"contentType": "application/xml"
				},
				"file" : "resources/catalog/text_analysis.xml"
			}
```
* * object->**name** - Name of the workflow.
* * object->**file** - Relative path of the XML file of the workflow.

4) Add the XML file(s) of the workflow(s) into `resources/catalog/` inside your package folder (e.g. `TextAnalysis/resources/catalog/text_analysis.xml`).

5) By default all new buckets will be added after all existing buckets inside catalog. So no need by default to add bucket name to `ordered_bucket_list` file.

But if you need to have strict order of buckets, then please update `ordered_bucket_list` by adding the package name (order by name). The whole list should be stored as 1 line without any spaces or end line character.

6) Update `build.gradle` by finding `task zip (type: Zip)` function and adding an **include** for your package (e.g. ``include 'TextAnalysis/**'``). Example:
```
task zip (type: Zip){
    archiveName="proactive-examples.zip"
    destinationDir = file('build/')
    from '.'
    include 'AWS/**'
    include 'Clearwater/**'
    include 'CloudAutomationTemplate/**'
    include 'Cron/**'
    include 'DockerBasics/**'
    include 'DockerSwarm/**'
    include 'Email/**'
    include 'FileFolderManagement/**'
    include 'FinanceMonteCarlo/**'
    include 'GetStarted/**'
    include 'HDFS/**'
    include 'ImageAnalysis/**'
    include 'JobAnalysis/**'
    include 'LogAnalysis/**'
    include 'MLBasics/**'
    include 'MLNodeSource/**'
    include 'OpenStack/**'
    include 'RemoteVisualization/**'
    include 'Spark/**'
    include 'SparkOrchestration/**'
    include 'Storm/**'
    include 'TextAnalysis/**'
    include 'TriggerTemplate/**'
    include 'TwitterApi/**'
    include 'WebNotification/**'
    include 'ordered_bucket_list'
}
```
#  The rules for added workflows
All the workflows added in proactive-examples project have to follow the next rules:

  . Every single workflow of packages distributed by Activeeon (as all the workflows from proactive-examples), MUST HAVE a Workflow Generic Information "workflow.icon" with a meaningful Icon
  . URL of this icon MUST reference a local file
  . If a workflow has a single task, this task MUST HAVE a Task Generic Information "task.icon" with the same icon as the Workflow

* _In case if workflow is not corresponding to specified rules: the test inside the proactive-examples project will fail._

That's all!
