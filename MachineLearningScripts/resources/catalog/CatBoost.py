# you can use the following search space to optimize the hyperparameters
# SEARCH_SPACE: {"kernel":choice(["rbf","sigmoid"])}

__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import json

input_variables = variables.get("INPUT_VARIABLES")
scoring = variables.get("SCORING")
alg_type = variables.get("TYPE")

algorithm = {
  'name': 'CatBoost',
  'type': alg_type.lower(),
  'is_supervised': True,
  'input_variables': input_variables,
  'scoring': scoring
}
print("algorithm: ", algorithm)

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)