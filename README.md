# ProActive Examples

## Quick start

Scripts in scripts/ are supposed to be triggered at scheduler start. For such integration,
update your script parameters if needed, and set your script paths into

    <PROACTIVE_HOME>/config/scheduler/settings.ini

following this

    pa.scheduler.startscripts.paths=/your/script1/path;/your/script2/path;/your/script3/path

For example, loadExamples.groovy has in charge the full deployment of an archive of workflows into a running ProActive installation.
