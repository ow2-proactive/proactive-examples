__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import json

algorithm = {
  'name': 'AutoSklearn_Regressor',
  'is_supervised': True,
  'type': 'regression',
  'task_time': int(variables.get("TASK_TIME")),
  'run_time': int(variables.get("RUN_TIME")),
  'sampling': variables.get("SAMPLING"),
  'sampling_strategy': variables.get('RESAMPLING_STRATEGY'),
  'folds': variables.get('FOLDS')
}

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)