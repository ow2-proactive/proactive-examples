print("BEGIN Import_Model")

import wget
import uuid
import shutil
import zipfile

from os.path import join, exists
from os import remove, listdir, makedirs

# Pre-trained models
# Text:  https://s3.eu-west-2.amazonaws.com/activeeon-public/models/basic_sentiment_analysis.zip
# Image: https://s3.eu-west-2.amazonaws.com/activeeon-public/models/model_resnet18.zip

# Load a trained model on ResNet-18 with two classes [ants, bees]
#MODEL_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/models/model_resnet18.zip' ##CLASSIFICATION MODEL

# Load a trained model on SegNet with five classes 
#MODEL_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/models/model_segnet.zip'##SEGMENTATION MODEL

# Load a trained model on SDD with twenty-one classes 
#MODEL_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/models/sdd_voc.zip'##OBJECT DETECTION (SSD MODEL ON PASCAL VOC DATASET)

# Load a trained model on SDD with twenty-one classes 
MODEL_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/models/yolo3_coco.zip'##OBJECT DETECTION (YOLO MODEL ON COCO DATASET)

if 'variables' in locals():
  MODEL_URL = variables.get("MODEL_URL")

print("MODEL_URL:   " + MODEL_URL)
assert MODEL_URL is not None

# Get an unique ID
ID = str(uuid.uuid4())

# Create an empty dir
MODEL_FOLDER = join('models', ID)
if exists(MODEL_FOLDER):
  shutil.rmtree(MODEL_FOLDER)
makedirs(MODEL_FOLDER)
print("MODEL_FOLDER: " + MODEL_FOLDER)

print("Downloading...")
filename = wget.download(MODEL_URL, MODEL_FOLDER)
print("FILENAME: " + filename)
print("OK")

print("Extracting...")
dataset_zip = zipfile.ZipFile(filename)
dataset_zip.extractall(MODEL_FOLDER)
dataset_zip.close()
remove(filename)
print("OK")

MODEL_PATH = None
LABELS_PATH = None
TEXT_PATH = None 
for file in listdir(MODEL_FOLDER):
  if file.endswith(".pt") or file.endswith(".pth") or file.endswith(".weights"):
    MODEL_PATH = join(MODEL_FOLDER, file)
  if file.endswith("label.pkl"):
    LABELS_PATH = join(MODEL_FOLDER, file)
  if file.endswith("text.pkl"):
    TEXT_PATH = join(MODEL_FOLDER, file)
  if file.endswith(".txt"):
    LABELS_PATH = join(MODEL_FOLDER, file)

assert MODEL_PATH is not None
#assert LABELS_PATH is not None
print(LABELS_PATH)
print("Model information: ")
print("MODEL_PATH:  " + MODEL_PATH)
if LABELS_PATH != None:
	print("LABELS_PATH: " + LABELS_PATH)
print("TEXT_PATH:   " + str(TEXT_PATH))

if 'variables' in locals():
  variables.put("MODEL_PATH", MODEL_PATH)
  variables.put("LABELS_PATH", LABELS_PATH)
  variables.put("TEXT_PATH", TEXT_PATH)
  variables.put("MODEL_FOLDER", MODEL_FOLDER)

print("END Import_Model")