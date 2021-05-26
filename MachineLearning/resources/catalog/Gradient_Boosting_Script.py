# Copyright Activeeon 2007-2021. All rights reserved.
# -------------------------------------------------------------
# You can use the following search space to optimize the hyperparameters
# SEARCH_SPACE: {"kernel": choice(["rbf", "sigmoid"])}

__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() == 'false':
    print("Task " + __file__ + " disabled")
    quit()

print("BEGIN " + __file__)

import json

input_variables = json.loads(variables.get("INPUT_VARIABLES"))
alg_type = str(variables.get("TYPE")).lower()
scoring = variables.get("SCORING")

algorithm = {
    'name': 'GradientBoosting',
    'type': alg_type,
    'is_supervised': True,
    'input_variables': input_variables,
    'scoring': scoring
}
print("algorithm:\n", algorithm)

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)