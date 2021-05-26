# Copyright Activeeon 2007-2021. All rights reserved.
print("BEGIN Download_Model")

import os
import uuid
import zipfile
import shutil

from os import remove, listdir, makedirs
from os.path import exists, join, isfile

if 'variables' in locals():
  MODEL_PATH  = variables.get("MODEL_PATH")
  MODEL_TYPE = variables.get("MODEL_TYPE")
  MODEL_ONNX_PATH  = variables.get("MODEL_ONNX_PATH")
  LABELS_PATH = variables.get("LABELS_PATH")
  TEXT_PATH   = variables.get("TEXT_PATH")

assert MODEL_PATH is not None

MODEL_TYPE = MODEL_TYPE.upper()

if MODEL_TYPE == 'ONNX' and MODEL_ONNX_PATH is not None:
	MODEL_PATH = MODEL_ONNX_PATH
elif MODEL_TYPE == 'ONNX' and MODEL_ONNX_PATH is None:
	print('This network does not yet support the ONNX format!')
elif MODEL_TYPE == 'PYTORCH':
	MODEL_PATH = MODEL_PATH
else:
	print('Please check the [MODEL_TYPE] parameter!')    
    
'''
assert MODEL_DIR_PATH is not None
assert exists(MODEL_DIR_PATH) == True

def zipdir(_path, _ziph):
  # ziph is zipfile handle
  for root, dirs, files in os.walk(_path):
    for file in files:
      _ziph.write(join(root, file))

zipf = zipfile.ZipFile('model.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir(MODEL_DIR_PATH, zipf)
zipf.close()
'''

# Get an unique ID
ID = str(uuid.uuid4())
FILE_NAME = ID + '.zip'

zipf = zipfile.ZipFile(FILE_NAME, 'w', zipfile.ZIP_DEFLATED)
zipf.write(MODEL_PATH)
if LABELS_PATH is not None:
  zipf.write(LABELS_PATH)
if TEXT_PATH is not None:
  zipf.write(TEXT_PATH)  
zipf.close()

assert isfile(FILE_NAME) == True

# Read the whole file at once
FILE_BIN = None
with open(FILE_NAME, "rb") as binary_file:
  FILE_BIN = binary_file.read()
assert FILE_BIN is not None

if 'variables' in locals():
  result = FILE_BIN
  resultMetadata.put("file.extension", ".zip")
  resultMetadata.put("file.name", "model.zip")
  resultMetadata.put("content.type", "application/octet-stream")

print("END Download_Model")