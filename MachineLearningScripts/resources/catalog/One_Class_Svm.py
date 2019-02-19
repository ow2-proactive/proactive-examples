__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import json

algorithm = {
  'name': 'OneClassSVM',
  'is_supervised': False,
  'type': 'anomaly',
  'kernel': variables.get("KERNEL"),
  'nu': float(variables.get("NU")),
  'gamma': float(variables.get("GAMMA"))
}

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)