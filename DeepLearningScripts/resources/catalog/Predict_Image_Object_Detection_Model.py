print("BEGIN Predict_Image_Object_Detection_Model")

import os
import cv2
import sys
import wget
import uuid
import glob
import torch
import numpy as np
import pandas as pd
from numpy import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn

from math import sqrt as sqrt
from skimage.transform import resize
from os import remove, listdir, makedirs
from itertools import product as product
from ast import literal_eval as make_tuple
from torch.utils.data import Dataset, DataLoader
from os.path import basename, splitext, exists, join

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

if 'variables' in locals():
  
  NUM_CLASSES = variables.get("NUM_CLASSES")
  NET_MODEL     = variables.get("NET_MODEL")
  NET_TRANSFORM = variables.get("NET_TRANSFORM")
  NET_CRITERION = variables.get("NET_CRITERION")
  DATASET_PATH  = variables.get("DATASET_PATH")
  MODEL_PATH     = variables.get("MODEL_PATH")
  LABEL_PATH  = variables.get("LABEL_PATH")
  IMG_SIZE = variables.get("IMG_SIZE")
  LEARNING_RATE = variables.get("LEARNING_RATE")
  MOMENTUM = float(str(variables.get("MOMENTUM")))    
  WEIGHT_DECAY = float(str(variables.get("WEIGHT_DECAY"))) 
  BATCH_SIZE = int(str(variables.get("BATCH_SIZE")))
  NUM_WORKERS = int(str(variables.get("NUM_WORKERS")))
  NET_NAME = variables.get("NET_NAME")

assert DATASET_PATH is not None
assert NET_MODEL is not None
assert NET_TRANSFORM is not None
assert MODEL_PATH is not None

IMG_SIZE = tuple(IMG_SIZE)

##  DOWNLOAD CLASSES FILE
print("Downloading...")
filename = wget.download(LABEL_PATH)
print("FILENAME: " + filename)
print("OK")

# Class names
CLASSES  = tuple(open(filename).read().splitlines())
num_pred = NUM_CLASSES

DATASET_TEST_PATH = join(DATASET_PATH, 'train')
loader = join(DATASET_TEST_PATH, 'images')

# http://pytorch.org/docs/master/cuda.html#torch.cuda.is_available
# Returns a bool indicating if CUDA is currently available.
use_gpu = torch.cuda.is_available()

# Get an unique ID
ID = str(uuid.uuid4())

# Create an empty dir
OUTPUT_FOLDER = join('output', ID)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
def random_color():
    levels = range(0,255)
    return tuple(random.choice(levels) for _ in range(3))

def genetate_color(num_pred):
    color_list = []
    for color in range(num_pred):
        color_list.append(random_color())
    return color_list

def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels

    
# ======================================================================================

##################################  BEGIN SSD NET ###################################### 

# ======================================================================================

if (NET_NAME == 'SSD'):

    USE_PRETRAINED_MODEL = variables.get("USE_PRETRAINED_MODEL")
    LR_STEPS = variables.get("LR_STEPS")
    MEANS = variables.get("MEANS")
    START_ITERATION  = int(str(variables.get("START_ITERATION")))
    MAX_ITERATION = int(str(variables.get("MAX_ITERATION")))
    MIN_SIZES = variables.get("MIN_SIZES")
    MAX_SIZES = variables.get("MAX_SIZES")
    BUILD_TYPE = 'test'
    LR_STEPS = tuple(LR_STEPS)
    MEANS = tuple(MEANS)
    
    MIN_SIZES  = make_tuple(MIN_SIZES)
    MIN_SIZES  = tuple(MIN_SIZES)
    MIN_SIZES  = list(MIN_SIZES)

    MAX_SIZES  = make_tuple(MAX_SIZES)
    MAX_SIZES  = tuple(MAX_SIZES)
    MAX_SIZES  = list(MAX_SIZES)

    # Load NET model
    exec(NET_MODEL)

    MODEL_NAME = 'SSD'
    ssd_net = build_ssd(BUILD_TYPE, IMG_SIZE[0], NUM_CLASSES)
    Net = ssd_net
    assert Net is not None, f'model {MODEL_NAME} not available'
    model = Net

    model.eval()

    # Load trained model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
    print('Finished loading model!')
 
    if use_gpu:
        model = model.cuda()
    
    # Load NET criterion
    exec(NET_CRITERION)

    # Load NET transform
    exec(NET_TRANSFORM)

    transform = BaseTransform(model.size, (104, 117, 123))

    def predict_model(_model, loader, _use_gpu, _num_pred):
        bbox_colors = []
        unique_labels = []
        image_name = []
        label_name = []
        color_list = genetate_color(_num_pred) 
    
        for img_paths in glob.glob(os.path.join(loader, "*")):
             img = cv2.imread(img_paths)
             img_name = os.path.basename(img_paths) # name file
             transform = BaseTransform(model.size, (104, 117, 123))
             x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
             x = Variable(x.unsqueeze(0))
             if use_gpu:
                 x = x.cuda()     
             y = model(x)      # forward 
             detections = y.data
    
             # scale each detection back up to the image
             scale = torch.Tensor([img.shape[1], img.shape[0],
                                     img.shape[1], img.shape[0]])    
            
             for i in range(detections.size(1)):
                 j = 0
                 while detections[0, i, j, 0] >= 0.6:
                        score = detections[0, i, j, 0]
                        pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                        coords = (pt[0], pt[1], pt[2], pt[3])             
                        x1 = pt[0]
                        y1 = pt[1]
                        box_w = pt[2]
                        box_h = pt[3]
                        class_name = CLASSES[i-1]
                        cls_pred  = i
                        print ('\t+ Label: %s, Conf: %.5f' % (class_name, score))
    
                        color = color_list[cls_pred]
                        color = tuple(map(int, color))
                       
                        cv2.rectangle(img , (x1, y1), (x1+box_w, y1+box_h), color, 6) 
                        cv2.putText(img, class_name, (pt[0], pt[1]), cv2.FONT_HERSHEY_COMPLEX, 1.5, color,2, cv2.LINE_AA)
                        j += 1 
                        
             label_paths = os.path.join(OUTPUT_FOLDER, img_name)
             cv2.imwrite(label_paths, img) 
             cv2.imwrite(os.path.join(OUTPUT_FOLDER) + '/' +  img_name, img)
            
             image_name.append(img_paths)
             label_name.append(label_paths)
        return image_name, label_name


    if os.path.isfile(filename):
        image_name, label_name = predict_model(model, loader, use_gpu, num_pred)
    else:
        print("Please, you need to add a class file")
        
# ======================================================================================

##################################  END SSD NET ######################################## 

# ======================================================================================




# ======================================================================================

##################################  BEGIN  YOLO  NET ###################################

# ======================================================================================

class ImageFolder(Dataset):
    def __init__(self, list_path, img_size=416):
        
        folder_path = join(list_path, 'images')
        
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)
        self.img_name  = os.listdir(folder_path)
        

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)
    

if (NET_NAME == 'YOLO'):
    
    CONF_THRESHOLD = variables.get("CONF_THRESHOLD")  
    NMS_THRESHOLD = variables.get("NMS_THRESHOLD")      
    
    ##  Download Model config
    print("Downloading...")
    MODEL_CONFIG_PATH = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/models/yolov3.cfg'
    filename = wget.download(MODEL_CONFIG_PATH)
    print("MODEL_CONFIG_PATH: " + filename)
    print("OK")
    model_config_path = os.path.realpath(filename)
    
    # Load NET transforms
    exec(NET_TRANSFORM)
    # Load NET model
    exec(NET_MODEL)

    Net = Darknet(model_config_path)
    model = Net

    # Load trained model
    model.load_weights(MODEL_PATH)
    
    model.eval()
    
    print('Finished loading model!')

    DATASET_TEST_PATH = join(DATASET_PATH, 'test')
    loader = DataLoader(ImageFolder(DATASET_TEST_PATH , IMG_SIZE[0]),num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=False)
    
    if use_gpu:
        Tensor = torch.cuda.FloatTensor
    Tensor = torch.FloatTensor
    
    def predict_model(_model, loader, _use_gpu):
           
        imgs = []           # Stores image paths
        img_detections = [] # Stores detections for each image index
        
        for batch_i, (img_paths, images) in enumerate(loader):
    
            if use_gpu:
                images = images.cuda()
            inputs = Variable(images.type(Tensor))
           
            # Get detections
            with torch.no_grad():
                outputs = model(inputs)
                outputs = non_max_suppression(outputs, NUM_CLASSES, CONF_THRESHOLD, NMS_THRESHOLD) 
                
            imgs.extend(img_paths)
            img_detections.extend(outputs) 
        return imgs, img_detections

    def img_labeled(imgs, img_detections, num_pred): 
        print ('\nSaving images:')
        
        color_list = genetate_color(num_pred)  
        image_name = []
        label_name = []
        
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)): 
            #print ("(%d) Image: '%s'" % (img_i, path))
            img = cv2.imread(path)         
            img_name = os.path.basename(path) 
           
            # The amount of padding that was added
            pad_x = max(img.shape[0] - img.shape[1], 0) * (IMG_SIZE[0] / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * (IMG_SIZE[0]/ max(img.shape))
            
            # Image height and width after padding is removed
            unpad_h = IMG_SIZE[0] - pad_y # check depois essa variavel para colocar a entrada resize
            unpad_w = IMG_SIZE[0] - pad_x    
               
            # Draw bounding boxes and labels of detections
            if detections is not None:
             
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:            
    
                    print ('\t+ Label: %s, Conf: %.5f' % (CLASSES[int(cls_pred)], cls_conf.item()))
        
                    # Rescale coordinates to original dimensions
                    box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                    box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                    y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                    x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]  
                    
                    id_label = CLASSES[int(cls_pred)]
                    findcolor = CLASSES.index(id_label)
                    color = color_list[int(findcolor)]
                    color = tuple(map(int, color))
                    
                    cv2.rectangle(img , (x1, y1), (x1+box_w, y1+box_h), color, 6)
                    cv2.putText(img, CLASSES[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1.5, color,2, cv2.LINE_AA)
                    cv2.imwrite(os.path.join(OUTPUT_FOLDER) + '/' + img_name, img)
                    
            else: 
               cv2.imwrite(os.path.join(OUTPUT_FOLDER) + '/' + img_name, img)
            
            label_paths = os.path.join(OUTPUT_FOLDER, img_name)
            image_name.append(path)
            label_name.append(label_paths)
        return image_name, label_name
                
    if os.path.isfile(filename):
        imgs, img_detections = predict_model(model, loader, use_gpu)
        image_name, label_name = img_labeled(imgs, img_detections, num_pred)
    else:
        print("Please, you need to add a class file")
        
df_name = pd.DataFrame(image_name)
df_image_name = pd.DataFrame(image_name)
df_label_name = pd.DataFrame(label_name)
df_name.columns = ['Image Paths']
df_image_name.columns = ['Images']
df_label_name.columns = ['Outputs']

df = pd.concat([df_name, df_image_name, df_label_name], axis=1)

if 'variables' in locals():
  variables.put("PREDICT_DATA_JSON", df.to_json(orient='split'))
  variables.put("BATCH_SIZE", BATCH_SIZE)
  variables.put("NUM_WORKERS", NUM_WORKERS)
  variables.put("OUTPUT_FOLDER", OUTPUT_FOLDER)

print("END Predict_Image_Object_Detection_Model")