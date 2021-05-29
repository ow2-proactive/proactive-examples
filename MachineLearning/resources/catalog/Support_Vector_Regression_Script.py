
# -------------------------------------------------------------
# You can use the following search space to optimize the hyperparameters
# SEARCH_SPACE: {"kernel":choice(["rbf", "sigmoid"])}

__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() == 'false':
    print("Task " + __file__ + " disabled")
    quit()

print("BEGIN " + __file__)

import json

input_variables = json.loads(variables.get("INPUT_VARIABLES"))
scoring = variables.get("SCORING")

algorithm = {
    'name': 'SupportVectorRegression',
    'is_supervised': True,
    'type': 'regression',
    'input_variables': input_variables,
    'scoring': scoring
}

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)