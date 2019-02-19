__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import json

algorithm = {
  'name': 'LogisticRegression',
  'is_supervised': True,
  'type': 'classification',
  'n_jobs': int(variables.get("N_JOBS")),
  'max_iter': int(variables.get("MAX_ITERATIONS")),
  'solver': variables.get("SOLVER"),
  'penalty': variables.get("PENALTY")
}

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)