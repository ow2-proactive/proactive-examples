# Copyright Activeeon 2007-2021. All rights reserved.
print("BEGIN Export_Images")

import os
import cv2
import json
import glob
import uuid  
import torch 
import numpy
import torch
import shutil
import random
import zipfile
import pandas as pd

from PIL import Image
from os.path import join, exists
from os import remove, listdir, makedirs

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset 

from torch.utils import model_zoo
from ast import literal_eval as make_tuple
from os.path import basename, splitext, exists, join, isfile

from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Scale

if 'variables' in locals():
  DATASET_PATH   = variables.get("DATASET_PATH")
  NET_TRANSFORM  = variables.get("NET_TRANSFORM")
  CNN_TRANSFORM  = variables.get("CNN_TRANSFORM")
  PREDICT_DATA   = variables.get("PREDICT_DATA_JSON")
  SHUFFLE = variables.get("SHUFFLE")
  BATCH_SIZE = int(str(variables.get("BATCH_SIZE")))
  NUM_WORKERS = int(str(variables.get("NUM_WORKERS")))
  DATASET_TYPE = variables.get("DATASET_TYPE")  
  IMG_SIZE   = variables.get("IMG_SIZE") 

#CLASSIFICATION
if DATASET_TYPE == 'CLASSIFICATION':
	# Load CNN transform
	# data_transforms
	if CNN_TRANSFORM != None:
		assert CNN_TRANSFORM is not None
		exec(CNN_TRANSFORM)   
        
	# Load dataset
	image_dataset = {x: 
		datasets.ImageFolder(join(DATASET_PATH, x), data_transforms[x]) 
  		for x in ['test']}

	data_loader = {x: 
		DataLoader(image_dataset[x], batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS) 
		for x in ['test']}        
        
	# Get an unique ID
	ID = str(uuid.uuid4())

	# Define localspace
	LOCALSPACE = join('results', ID)
	os.makedirs(LOCALSPACE, exist_ok=True)
    
	if PREDICT_DATA != None: 
		prediction_result  = pd.read_json(PREDICT_DATA, orient='split')
		df = pd.DataFrame(prediction_result)
		preds = df['Predictions']
        
		for index, elem in enumerate(preds):
			# check if folder exist
			os.makedirs(join(LOCALSPACE,  preds[index]), exist_ok=True)       
			shutil.copy2(data_loader['test'].dataset.imgs[index][0], LOCALSPACE + '/' + preds[index])
            
		FILE_NAME = '.zip'  
		FILE_PATH = join(LOCALSPACE, FILE_NAME)
		print("FILE_PATH: " + FILE_PATH)  
             
		def zipdir(_path, _ziph):
			# ziph is zipfile handle
			for root, dirs, files in os.walk(_path):
				for file in files:
					_ziph.write(join(root, file))   
            
		zipf = zipfile.ZipFile(FILE_PATH, 'w', zipfile.ZIP_DEFLATED)
		zipdir(LOCALSPACE, zipf)
		zipf.close()
  
		assert isfile(FILE_PATH) == True   
        
		# Read the whole file at once
		FILE_BIN = None
		with open(FILE_PATH, "rb") as binary_file:
			FILE_BIN = binary_file.read()
		assert FILE_BIN is not None  
               
		if 'variables' in locals():
			result = FILE_BIN
			resultMetadata.put("file.extension", ".zip")
			resultMetadata.put("file.name", "result.zip")
			resultMetadata.put("content.type", "application/octet-stream") 
			print("END Export_Images")
	else:
		print("It is not possible to export the images")   
        
#DETECTION / SEGMENTATION     
if DATASET_TYPE == 'DETECTION' or DATASET_TYPE == 'SEGMENTATION':
    IMG_SIZE = tuple(IMG_SIZE)
    # Get an unique ID
    ID = str(uuid.uuid4())
    # Define localspace
    LOCALSPACE = join('results', ID)
    os.makedirs(LOCALSPACE, exist_ok=True)
    if PREDICT_DATA != None: 
        prediction_result  = pd.read_json(PREDICT_DATA, orient='split')
        df = pd.DataFrame(prediction_result)
        os.makedirs(join(LOCALSPACE, 'images'), exist_ok=True)
        os.makedirs(join(LOCALSPACE, 'outputs'), exist_ok=True)
        imgs = df['Images']        
        preds = df['Outputs']
        for index, elem in enumerate(preds): 
        	shutil.copy2(imgs[index], LOCALSPACE + '/' + 'images') 
        	shutil.copy2(elem, LOCALSPACE + '/' + 'outputs') 
            
        FILE_NAME = '.zip' 
        FILE_PATH = join(LOCALSPACE, FILE_NAME)
        print("FILE_PATH: " + FILE_PATH) 
        
        def zipdir(_path, _ziph):
        	# ziph is zipfile handle
        	for root, dirs, files in os.walk(_path):
        		for file in files:
        			_ziph.write(join(root, file))  
                    
        zipf = zipfile.ZipFile(FILE_PATH, 'w', zipfile.ZIP_DEFLATED)
        zipdir(LOCALSPACE, zipf)
        zipf.close()
        
        assert isfile(FILE_PATH) == True 
        
		# Read the whole file at once
        FILE_BIN = None
        with open(FILE_PATH, "rb") as binary_file:
        	FILE_BIN = binary_file.read()
        assert FILE_BIN is not None  
        
        if 'variables' in locals():
        	result = FILE_BIN
        	resultMetadata.put("file.extension", ".zip")
        	resultMetadata.put("file.name", "result.zip")
        	resultMetadata.put("content.type", "application/octet-stream") 
        	print("END Export_Images")
    else:
        print("It is not possible to export the images")