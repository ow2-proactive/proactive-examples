__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import json

algorithm = {
  'name': 'SupportVectorRegression',
  'is_supervised': True,
  'type': 'regression',
  'C': float(variables.get("C")),
  'kernel': variables.get("KERNEL"),
  'epsilon': float(variables.get("EPSILON"))
}

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)