
print("BEGIN Train_Image_Classification_Model")

import os
import ast
import time
import copy
import uuid
import json
import argparse
import numpy as np  
from PIL import Image

from os.path import join, exists
from os import remove, listdir, makedirs

import torch
import torch.nn as nn
import torch.optim as optim

import torch.onnx
import torchvision

from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

VISDOM_ENABLED = variables.get("ENDPOINT_VISDOM")

if VISDOM_ENABLED is not None:
  from visdom import Visdom

if 'variables' in locals():
  if variables.get("NUM_EPOCHS") is not None:
    NUM_EPOCHS = int(str(variables.get("NUM_EPOCHS")))
  DATASET_PATH  = variables.get("DATASET_PATH")
  CNN_MODEL     = variables.get("CNN_MODEL")
  CNN_TRANSFORM = variables.get("CNN_TRANSFORM")
  SHUFFLE = variables.get("SHUFFLE")
  MODEL_TYPE = variables.get("MODEL_TYPE")
  BATCH_SIZE = int(str(variables.get("BATCH_SIZE")))
  NUM_WORKERS = int(str(variables.get("NUM_WORKERS")))

assert DATASET_PATH is not None
assert CNN_MODEL is not None
assert CNN_TRANSFORM is not None

# save onnx model
MODEL_ONNX_TYPE = False

# Load CNN transform
# data_transforms
exec(CNN_TRANSFORM)

# Load dataset
image_datasets = {x: 
  datasets.ImageFolder(join(DATASET_PATH, x), data_transforms[x]) 
  for x in ['train', 'val']}

data_loaders = {x: 
  DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS) 
  for x in ['train', 'val']}

# Training data is required
assert len(image_datasets['train']) > 0

# If validation set is empty, use the trainning set
if len(image_datasets['val']) == 0:
  image_datasets['val'] = image_datasets['train']
  data_loaders['val'] = data_loaders['train']

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

# Load CNN model
exec(CNN_MODEL)

# http://pytorch.org/docs/master/cuda.html#torch.cuda.is_available
# Returns a bool indicating if CUDA is currently available.
use_gpu = torch.cuda.is_available()
if use_gpu:
  cnn = cnn.cuda()

# http://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss
# This criterion combines LogSoftMax and NLLLoss in one single class.
criterion = nn.CrossEntropyLoss()

# http://pytorch.org/docs/master/optim.html#torch.optim.SGD
# Implements stochastic gradient descent (optionally with momentum).
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

# http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.StepLR
# Sets the learning rate of each parameter group to the initial lr decayed by gamma every step_size epochs. 
# When last_epoch=-1, sets initial lr as lr.
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


###############################IF VISDOM IS ENABLED###############################

if VISDOM_ENABLED is not None:
  visdom_endpoint = VISDOM_ENABLED.replace("http://", "")

  (VISDOM_HOST, VISDOM_PORT) = visdom_endpoint.split(":")

  print("Connecting to %s" % VISDOM_PORT)
  viz = Visdom(server="http://"+VISDOM_HOST, port=VISDOM_PORT)
  assert viz.check_connection()


  win_global_loss_train = viz.line(Y = np.array([1]), X = np.array([1]),
                             opts = dict(
                                    xlabel = 'Epoch',
                                    ylabel = 'Loss',
                                    title = 'Training loss (per epoch)',
                                    ),
                             )


  win_global_acc_train = viz.line(Y = np.array([1]), X = np.array([1]),
                             opts = dict(
                                    xlabel = 'Epoch',
                                    ylabel = 'Accuracy',
                                    title = 'Training accuracy (per epoch)',
                                    ),
                             )                                   


  win_train = viz.text("Training:\n")  
                                  
                                  
  win_global_loss_val = viz.line(Y = np.array([1]), X = np.array([1]),
                             opts = dict(
                                    xlabel = 'Epoch',
                                    ylabel = 'Loss',
                                    title = 'Validation loss (per epoch)',
                                    ),
                             )


  win_global_acc_val = viz.line(Y = np.array([1]), X = np.array([1]),
                                    opts = dict(
                                    xlabel = 'Epoch',
                                    ylabel = 'Accuracy',
                                    title = 'Validation accuracy (per epoch)',
                                    ),
                             )                                   
                                 
  win_val = viz.text("Validation:\n")

##################################################################################
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        scheduler.step()
        model.train(True)  # Set model to training mode
      else:
        model.train(False)  # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for data in data_loaders[phase]:
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
          inputs = Variable(inputs.cuda())
          labels = Variable(labels.cuda())
        else:
          inputs, labels = Variable(inputs), Variable(labels)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        if phase == 'train':
          loss.backward()
          optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.item() / dataset_sizes[phase]

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


	  ##################################################IF VISDOM IS ENABLED###########################################
      if VISDOM_ENABLED is not None:
          if phase == 'train':
              viz.text('-' * 10, win=win_train, append=True)  
              viz.text('Epoch {}/{}'.format(epoch, num_epochs - 1), win=win_train, append=True)            
              viz.text('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc), win=win_train, append=True)
          else:
              viz.text('-' * 10, win=win_val, append=True)         
              viz.text('Epoch {}/{}'.format(epoch, num_epochs - 1), win=win_val, append=True)    
              viz.text('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc), win=win_val, append=True)          

          # plot loss and accuracy per epoch
          if epoch == 1 and phase == 'train':
              viz.line(Y = np.array([epoch_loss]), X = np.array([epoch]), win = win_global_loss_train, update='replace')
              viz.line(Y = np.array([epoch_acc]), X = np.array([epoch]), win = win_global_acc_train, update='replace')            
          elif epoch != 1 and phase == 'train':
              viz.line(Y = np.array([epoch_loss]), X = np.array([epoch]), win = win_global_loss_train, update='append')
              viz.line(Y = np.array([epoch_acc]), X = np.array([epoch]), win = win_global_acc_train, update='append')        

          # plot loss and accuracy per epoch
          if epoch == 1 and phase == 'val':
              viz.line(Y = np.array([epoch_loss]), X = np.array([epoch]), win = win_global_loss_val, update='replace')
              viz.line(Y = np.array([epoch_acc]), X = np.array([epoch]), win = win_global_acc_val, update='replace')            
          elif epoch != 1 and phase == 'val':
              viz.line(Y = np.array([epoch_loss]), X = np.array([epoch]), win = win_global_loss_val, update='append')
              viz.line(Y = np.array([epoch_acc]), X = np.array([epoch]), win = win_global_acc_val, update='append')              
	  ####################################################################################################################
      
      
      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
      
    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model

# Return the best model
best_cnn = train_model(cnn, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

# Get an unique ID
ID = str(uuid.uuid4())

# Create an empty dir
MODEL_FOLDER = join('models', ID)
os.makedirs(MODEL_FOLDER, exist_ok=True)
print("MODEL_FOLDER: " + MODEL_FOLDER)

print('Saving trained model...')
MODEL_PATH = join(MODEL_FOLDER, "model.pt")
torch.save(best_cnn, MODEL_PATH)

# Save onnx trained model
if MODEL_ONNX_TYPE:
  print('Saving trained model...')
  MODEL_ONNX_PATH = join(MODEL_FOLDER, "model.onnx")
  CHANNEL = 3
  #input - 3 channels, 224x224 image size,
  dummy_input = Variable(torch.randn(BATCH_SIZE, CHANNEL, 224, 224))
  # Invoke export
  torch.onnx.export(best_cnn, dummy_input, MODEL_ONNX_PATH)
else:
  MODEL_ONNX_PATH  = None
    
# Save labels
print('Saving labels to a text file...')
LABELS_PATH = join(MODEL_FOLDER, "labels.txt")
with open(LABELS_PATH, 'w') as outfile:
  json.dump(class_names, outfile)

print("Model information: ")
print("MODEL_PATH:  " + MODEL_PATH)
print("LABELS_PATH: " + LABELS_PATH)

if 'variables' in locals():
  variables.put("MODEL_FOLDER", MODEL_FOLDER)
  variables.put("MODEL_PATH", MODEL_PATH)
  variables.put("MODEL_ONNX_PATH", MODEL_ONNX_PATH)
  variables.put("LABELS_PATH", LABELS_PATH)
  variables.put("SHUFFLE", SHUFFLE)
  variables.put("BATCH_SIZE", BATCH_SIZE)
  variables.put("NUM_WORKERS", NUM_WORKERS)

print("END Train_Image_Classification_Model")