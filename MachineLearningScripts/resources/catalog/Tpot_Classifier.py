__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import json

algorithm = {
  'name': 'TPOT_Classifier',
  'is_supervised': True,
  'automl': False,
  'type': 'classification',
  'generations': int(variables.get("GENERATIONS")),
  'cv': int(variables.get("CV")),
  'scoring': variables.get("SCORING"),
  'verbosity': int(variables.get("VERBOSITY"))
}

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)