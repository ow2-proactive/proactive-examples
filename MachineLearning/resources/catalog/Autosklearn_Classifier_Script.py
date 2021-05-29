
__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() == 'false':
    print("Task " + __file__ + " disabled")
    quit()

print("BEGIN " + __file__)

import json
from distutils.util import strtobool

algorithm = {
    'name': 'AutoSklearn_Classifier',
    'is_supervised': True,
    'type': 'classification',
    'automl': False,
    'task_time': int(variables.get("TASK_TIME")),
    'run_time': int(variables.get("RUN_TIME")),
    'sampling': bool(strtobool(variables.get("SAMPLING"))),
    'sampling_strategy': variables.get('SAMPLING_STRATEGY'),
    'folds': int(variables.get('FOLDS'))
}
print("algorithm:\n", algorithm)

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)