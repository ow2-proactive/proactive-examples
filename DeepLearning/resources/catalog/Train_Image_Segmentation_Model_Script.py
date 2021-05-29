
print("BEGIN Train_Image_Segmentation_Model")

import os, time, copy, uuid, json, argparse
import numpy as np 
from PIL import Image
 
from os.path import join, exists
from os import remove, listdir, makedirs
from ast import literal_eval as make_tuple

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset 

from torchvision import models
from torch.utils import model_zoo
  
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Resize

if 'variables' in locals():
    NUM_EPOCHS    = int(str(variables.get("NUM_EPOCHS")))
    NUM_CLASSES   = variables.get("NUM_CLASSES")
    DATASET_PATH  = variables.get("DATASET_PATH")
    NET_MODEL     = variables.get("NET_MODEL")
    NET_TRANSFORM = variables.get("NET_TRANSFORM")
    NET_CRITERION = variables.get("NET_CRITERION") 
    IMG_SIZE      = variables.get("IMG_SIZE")
    SHUFFLE       = variables.get("SHUFFLE")
    BATCH_SIZE    = int(str(variables.get("BATCH_SIZE")))
    NUM_WORKERS   = int(str(variables.get("NUM_WORKERS")))

assert DATASET_PATH is not None
assert NET_MODEL is not None
assert NET_TRANSFORM is not None
assert NET_CRITERION is not None

IMG_SIZE = tuple(IMG_SIZE)

# save onnx model
MODEL_ONNX_TYPE = False
MODEL_ONNX_PATH = None
if MODEL_ONNX_TYPE:
    print('This network does not support the ONNX format yet!')

# Load NET model
exec(NET_MODEL)

# Load CNN transform
# data_transforms
exec(NET_TRANSFORM)

# VOC12 DATASET
EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):
    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'classes')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
            image_size = image.size
        
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            if NUM_CLASSES == 1:
                label = load_image(f).convert('L')
            else:
                label = load_image(f).convert('P')
            file_name = image_path(self.labels_root, filename, '.png')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, image_size, file_name

    def __len__(self):
        return len(self.filenames)


# Load train dataset
DATASET_TRAIN_PATH = join(DATASET_PATH, 'train')
loader = DataLoader(VOC12(DATASET_TRAIN_PATH, input_transform, target_transform), 
                    num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)

# http://pytorch.org/docs/master/cuda.html#torch.cuda.is_available
# Returns a bool indicating if CUDA is currently available.
use_gpu = torch.cuda.is_available()

if use_gpu:
    model = model.cuda()
model.train()

weight = torch.ones(NUM_CLASSES)
weight[0] = 0

# Load NET criterion
exec(NET_CRITERION)

if NUM_CLASSES == 1:
    if use_gpu:
        criterion = BinaryCrossEntropyLoss2d()
    else:
        criterion = BinaryCrossEntropyLoss2d()
    # Observe that all parameters are being optimized
    optimizer_ft =SGD(model.parameters(), lr=.1, momentum=.9) 
else:
    if use_gpu:
        criterion = CrossEntropyLoss2d(weight.cuda())
    else:
        criterion = CrossEntropyLoss2d(weight)
    # Observe that all parameters are being optimized
    optimizer_ft = SGD(model.parameters(), 1e-3, .9)


############################## IF VISDOM IS ENABLED ##############################
VISDOM_ENABLED = variables.get("ENDPOINT_VISDOM")

if VISDOM_ENABLED is not None:
    from visdom import Visdom

    visdom_endpoint = VISDOM_ENABLED.replace("http://", "")

    (VISDOM_HOST, VISDOM_PORT) = visdom_endpoint.split(":")  

    print("Connecting to %s" % VISDOM_PORT)
    viz = Visdom(server="http://"+VISDOM_HOST, port=VISDOM_PORT)
    assert viz.check_connection()

    win_train = viz.text("Training:\n")  

    win_global_loss_train = viz.line(Y=np.array([np.nan]), X=np.array([np.nan]),
                               opts=dict(
                                      xlabel='Epoch',
                                      ylabel='Loss',
                                      title='Model loss (per epoch)',
                                      )
                               )

    win_global_avgloss_train = viz.line(Y=np.array([np.nan]), X=np.array([np.nan]),
                               opts=dict(
                                      xlabel='Epoch',
                                      ylabel='Avg loss',
                                      title='Model avg loss (per epoch)',
                                      )
                               )
##################################################################################

def train_model(model, criterion, optimizer, num_epochs):
    since = time.time()   
    best_model = copy.deepcopy(model.state_dict())
    best_loss = None
    for epoch in range(1, NUM_EPOCHS+1):
        epoch_loss = []
        print('Epoch {}/{}'.format(epoch, NUM_EPOCHS))
        print('-' * 10)
       
        for step, (images, labels, image_size, filename) in enumerate(loader):
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)
            
            optimizer.zero_grad()
            
            if NUM_CLASSES == 1:
                loss = criterion(outputs.view(-1), targets.view(-1))
            else:
                loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
            avg_loss = sum(epoch_loss) / len(epoch_loss)
            
            #print(f'loss: {avg_loss} (epoch: {epoch}, step: {step})')
        print(f'epoch: {epoch}, avg loss: {avg_loss}, loss: {loss.item()}')

################################################## IF VISDOM IS ENABLED ###########################################
        if VISDOM_ENABLED is not None:
            viz.text('-' * 30, win=win_train, append=True)
            viz.text('Epoch {}/{}'.format(epoch, NUM_EPOCHS), win=win_train, append=True)
            viz.text('Loss: {:.4f} Avg: {:.4f}'.format(loss.item(), avg_loss), win=win_train, append=True)
            
            # plot loss and accuracy per epoch
            viz.line(Y=np.array([loss.item()]), X=np.array([epoch]), win=win_global_loss_train, update='append')
            viz.line(Y=np.array([avg_loss]), X=np.array([epoch]), win=win_global_avgloss_train, update='append')
####################################################################################################################

        if best_loss is None or loss.item() < best_loss:
            best_loss = loss.item()
            best_model = copy.deepcopy(model.state_dict())
    print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best model loss: {:.4f}'.format(best_loss))
    model.load_state_dict(best_model)
    return model

# Return the best model
best_model = train_model(model, criterion, optimizer_ft, num_epochs=NUM_EPOCHS)

# Get an unique ID
ID = str(uuid.uuid4())

# Create an empty dir
MODEL_FOLDER = join('models', ID)
os.makedirs(MODEL_FOLDER, exist_ok=True)
print("MODEL_FOLDER: " + MODEL_FOLDER)

# Save pytorch trained model
print('Saving trained model...')
MODEL_PATH = join(MODEL_FOLDER, "model.pt")
torch.save(best_model, MODEL_PATH)
print("Model information: ")
print("MODEL_PATH:  " + MODEL_PATH)

if 'variables' in locals():
    variables.put("MODEL_FOLDER", MODEL_FOLDER)
    variables.put("MODEL_PATH", MODEL_PATH)
    variables.put("MODEL_ONNX_PATH", MODEL_ONNX_PATH)
    variables.put("SHUFFLE", SHUFFLE)
    variables.put("BATCH_SIZE", BATCH_SIZE)
    variables.put("NUM_WORKERS", NUM_WORKERS)
    variables.put("NUM_CLASSES", NUM_CLASSES)

print("END Train_Image_Segmentation_Model")