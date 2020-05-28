__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() == 'false':
    print("Task " + __file__ + " disabled")
    quit()

print("BEGIN " + __file__)

import json

algorithm = {
    'name': 'TPOT_Regressor',
    'is_supervised': True,
    'type': 'regression',
    'automl': False,
    'generations': int(variables.get("GENERATIONS")),
    'cv': int(variables.get("CV")),
    'scoring': variables.get("SCORING"),
    'verbosity': int(variables.get("VERBOSITY"))
}
print("algorithm:\n", algorithm)

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)
