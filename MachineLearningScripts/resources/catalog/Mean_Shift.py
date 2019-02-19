__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import json

algorithm = {
  'name': 'MeanShift',
  'is_supervised': False,
  'type': 'clustering',
  'cluster_all': variables.get("CLUSTER_ALL"),
  'n_jobs': int(variables.get("N_JOBS"))
}

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)