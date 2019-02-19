__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import json

algorithm = {
  'name': 'KMeans',
  'is_supervised': False,
  'type': 'clustering',
  'n_clusters': int(variables.get("N_CLUSTERS")),
  'n_jobs': int(variables.get("N_JOBS")),
  'max_iterations': int(variables.get("MAX_ITERATIONS"))
}

algorithm_json = json.dumps(algorithm)
resultMetadata.put("task.algorithm_json", algorithm_json)

print("END " + __file__)