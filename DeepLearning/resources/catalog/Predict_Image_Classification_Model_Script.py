# Copyright Activeeon 2007-2021. All rights reserved.
print("BEGIN Predict_Image_Classification_Model")

import os
import torch
import json
import numpy as np
import pandas as pd
from os.path import join
from torch.autograd import Variable
import xml.sax.saxutils as saxutils 
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

if 'variables' in locals():
  MODEL_PATH     = variables.get("MODEL_PATH")
  DATASET_PATH   = variables.get("DATASET_PATH")
  SHUFFLE = variables.get("SHUFFLE")
  BATCH_SIZE = int(str(variables.get("BATCH_SIZE")))
  NUM_WORKERS = int(str(variables.get("NUM_WORKERS")))
  LABELS_PATH    = variables.get("LABELS_PATH")
  CNN_TRANSFORM  = variables.get("CNN_TRANSFORM")

assert MODEL_PATH is not None
assert DATASET_PATH is not None
assert LABELS_PATH is not None
assert CNN_TRANSFORM is not None

class_names = None
with open(LABELS_PATH, 'r') as f:
  class_names = json.load(f)
assert class_names is not None

# Load trained model
model=torch.load(MODEL_PATH,map_location={'cuda:0': 'cpu'})

# http://pytorch.org/docs/master/cuda.html#torch.cuda.is_available
# Returns a bool indicating if CUDA is currently available.
use_gpu = torch.cuda.is_available()
if use_gpu:
  model = model.cuda()

# Load CNN transform
# data_transforms
exec(CNN_TRANSFORM)

# Load dataset
image_dataset = {x: 
  datasets.ImageFolder(join(DATASET_PATH, x), data_transforms[x]) 
  for x in ['test']}

data_loader = {x: 
  DataLoader(image_dataset[x], batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS) 
  for x in ['test']}

def predict_model(_model, _data_loader, _use_gpu, _class_names, _max_images=None):
  images_so_far = 0
  predictions = []
  image_target = []  
  for i, data in enumerate(_data_loader['test']):
    inputs, labels = data
    
    if _use_gpu:
      inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
      inputs, labels = Variable(inputs), Variable(labels)

    outputs = _model(inputs)
    _, preds = torch.max(outputs.data, 1)
    print(preds)

    for j in range(inputs.size()[0]):
      images_so_far += 1
      result = _class_names[preds[j]]
      predictions.append(result)
      image_target.append(inputs)
      if images_so_far == _max_images:
        return
    
    #break
  return predictions, image_target 

preds, image_target = predict_model(model, data_loader, use_gpu, class_names)

label_test = []
image_input = []
image_name = []
for index, elem in enumerate(image_target):
  input_select = data_loader['test'].dataset.imgs[index]
  dir_image_temp = input_select[0]
  id_label = input_select[1]  
  com_data_name = os.path.basename(dir_image_temp)    
  lab_d = data_loader['test'].dataset.classes
  lab_d = list(lab_d)
  label = lab_d[id_label]
  label_test.append(label)
  image_input.append(dir_image_temp)
  image_name.append(com_data_name)

reponse_good = '&#9989;'
reponse_bad = '&#10060;'

df_preds = pd.DataFrame(preds)
df_label = pd.DataFrame(label_test)
df_name = pd.DataFrame(image_name)
df_preds.columns = ['Predictions']
df_label.columns = ['Targets']
df_name.columns = ['Image Names']
df_test_image = pd.DataFrame(image_input)
df_test_image.columns = ['Images']
df = pd.concat([df_test_image, df_name, df_label, df_preds], axis=1)
df['Results'] = np.where((df['Predictions'] == df['Targets']), saxutils.unescape(reponse_good), saxutils.unescape(reponse_bad)) 
    
if 'variables' in locals():
  variables.put("PREDICT_DATA_JSON", df.to_json(orient='split'))
  variables.put("BATCH_SIZE", BATCH_SIZE)
  variables.put("NUM_WORKERS", NUM_WORKERS)
  variables.put("SHUFFLE", SHUFFLE)

print("END Predict_Image_Classification_Model")