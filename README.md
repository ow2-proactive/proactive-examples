# ProActive Examples

This repository includes a sub directory per bucket, in which a metadata file centralizes object-related-informations: bucket name, object name, object kind,..
If the "studio_template" section is specified, the object/workflow is also considered as a studio template.

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
					"kind": "workflow",
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

* * catalog->**bucket** - Set the name of the bucket for this package.
* * catalog->**objects** - Add an object of each workflow of the package.

An example of a catalog object that represents a workflow:
```
{
				"name" : "text_analysis",
				"metadata" : {
					"kind": "workflow",
					"commitMessage": "First commit",
					"contentType": "application/xml"
				},
				"file" : "resources/catalog/text_analysis.xml"
			}
```
* * object->**name** - Name of the workflow.
* * object->**file** - Relative path of the XML file of the workflow.

4) Add the XML file(s) of the workflow(s) into `resources/catalog/` inside your package folder (e.g. `TextAnalysis/resources/catalog/text_analysis.xml`).

5) Update `ordered_bucket_list` by adding the package name (order by name).

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
    include 'HDFSOrchestration/**'
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

That's all!