__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import json

algorithm = {
  'name': 'IsolationForest',
  'is_supervised': False,
  'type': 'anomaly',
  'n_jobs': int(variables.get("N_JOBS")),
  'n_estimators': int(variables.get("N_ESTIMATORS"))
}

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)