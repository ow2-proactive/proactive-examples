__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import os, sys, bz2, uuid, wget, pickle

MODEL_URL = variables.get("MODEL_URL")
assert MODEL_URL is not None and MODEL_URL is not ""

filename = wget.download(str(MODEL_URL))
model = pickle.load(open(filename, "rb"))

model_bin = pickle.dumps(model)
assert model_bin is not None

compressed_model = bz2.compress(model_bin)

model_id = str(uuid.uuid4())
variables.put(model_id, compressed_model)

print("model id: ", model_id)
print('model size (original):   ', sys.getsizeof(model_bin), " bytes")
print('model size (compressed): ', sys.getsizeof(compressed_model), " bytes")

resultMetadata.put("task.name", __file__)
resultMetadata.put("task.model_id", model_id)

#result = model_bin
#resultMetadata.put("file.extension", ".model")
#resultMetadata.put("file.name", "myModel.model")
#resultMetadata.put("content.type", "application/octet-stream")

print("END " + __file__)