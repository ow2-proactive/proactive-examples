__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import json

algorithm = {
  'name': 'BayesianRidgeRegression',
  'is_supervised': True,
  'type': 'regression',
  'alpha_1': float(variables.get("ALPHA_1")),
  'alpha_2': float(variables.get("ALPHA_2")),
  'lambda_1': float(variables.get("LAMBDA_1")),
  'lambda_2': float(variables.get("LAMBDA_2")),
  'n_iter': int(variables.get("N_ITERATIONS"))
}

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)