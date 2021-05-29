
# -------------------------------------------------------------
# You can use the following search space to optimize the hyperparameters
# SEARCH SPACE: {
# "max_iter": choice([5, 10, 15]),
# "solver": choice(["liblinear", "saga"]),
# "penalty": choice(["l1", "l2"])
# }

__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() == 'false':
    print("Task " + __file__ + " disabled")
    quit()

print("BEGIN " + __file__)

import json

input_variables = json.loads(variables.get("INPUT_VARIABLES"))
scoring = variables.get("SCORING")

algorithm = {
    'name': 'LogisticRegression',
    'type': 'classification',
    'is_supervised': True,
    'input_variables': input_variables,
    'scoring': scoring
}
print("algorithm:\n", algorithm)

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)