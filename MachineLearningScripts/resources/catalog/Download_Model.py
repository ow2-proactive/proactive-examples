__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import os, sys, bz2

input_variables = {'task.model_id': None}
for key in input_variables.keys():
  for res in results:
    value = res.getMetadata().get(key)
    if value is not None:
      input_variables[key] = value
      break

model_id = input_variables['task.model_id']
model_compressed = variables.get(model_id)
model_bin = bz2.decompress(model_compressed)
assert model_bin is not None

print("model id (in): ", model_id)
print("model size: ", sys.getsizeof(model_compressed), " bytes")
print("model size (decompressed): ", sys.getsizeof(model_bin), " bytes")

assert model_bin is not None
result = model_bin

#resultMetadata.put("task.name", __file__)
#resultMetadata.put("task.model_bin", model_bin)

resultMetadata.put("file.extension", ".model")
resultMetadata.put("file.name", "myModel.model")
resultMetadata.put("content.type", "application/octet-stream")

print("END " + __file__)