print("BEGIN Model_Explainability")

import matplotlib as mpl
mpl.use('Agg')
import os
import io
import cv2
import json
import glob
import shap
import torch
import shutil
import base64
import warnings
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from os.path import join
import torch, torchvision
import matplotlib.pyplot as pl
import xml.sax.saxutils as saxutils 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from os import remove, listdir, makedirs
from distutils.dir_util import copy_tree
from os.path import basename, splitext, exists, join
from torchvision import transforms, models, datasets
from matplotlib.colors import LinearSegmentedColormap 


if 'variables' in locals():
  MODEL_PATH     = variables.get("MODEL_PATH")
  DATASET_PATH   = variables.get("DATASET_PATH")
  LABELS_PATH    = variables.get("LABELS_PATH")
  CNN_TRANSFORM  = variables.get("CNN_TRANSFORM")
  X_IMG_LIST = variables.get("IMG_LIST")
  IMG_LIST = [x.strip() for x in X_IMG_LIST.split(',')]
  IMG_SAMPLES = int(str(variables.get("IMG_SAMPLES")))
  FEATURE_LAYER = variables.get("FEATURE_LAYER")
  RANKED_OUTPUTS = int(str(variables.get("RANKED_OUTPUTS")))    
    
assert MODEL_PATH is not None
assert DATASET_PATH is not None
assert LABELS_PATH is not None
assert CNN_TRANSFORM is not None
assert IMG_LIST is not None
assert IMG_SAMPLES is not None
assert FEATURE_LAYER is not None


class_names = None
with open(LABELS_PATH, 'r') as f:
  class_names = json.load(f)
assert class_names is not None

NEW_PATH = DATASET_PATH + '/' + 'new_data'

# Load trained model
model=torch.load(MODEL_PATH,map_location={'cuda:0': 'cpu'})


# http://pytorch.org/docs/master/cuda.html#torch.cuda.is_available
# Returns a bool indicating if CUDA is currently available.
use_gpu = torch.cuda.is_available()
if use_gpu:
  model = model.cuda()


# new dataset with all images
def new_dataset(DATASET_PATH):
  
    images_list = []
    
    os.makedirs(NEW_PATH, exist_ok=True)
    for root in listdir(DATASET_PATH + '/' + 'test'):
        if (not root.startswith('.')):
            print(root)
            images_list = join(DATASET_PATH + '/' + 'test', root)
            copy_tree(images_list, NEW_PATH)

# load new dataset
def load_new_dataset(DATASET_PATH):

    image_list = []
    
    for filename in glob.glob(NEW_PATH + '/' + '*jpg'):
        print(filename)
        img = cv2.imread(filename)
        height, width, channels = img.shape 
        img_res = cv2.resize(img,(height, width))
        image_list.append(cv2.resize(cv2.imread(filename),(224, 224))) ### resolucao pegar automaticamente

    X =  np.array(image_list).astype(np.float32)    
    return X


def normalize(image):
    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    # in addition, roll the axis so that they suit pytorch
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()


new_dataset(DATASET_PATH)
X = load_new_dataset(DATASET_PATH) 
X /= 255

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
    
to_explain = X[[int(IMG_LIST[0]), int(IMG_LIST[1])]] # choose 2 images 


# features_layer=model.features[7]
exec("features_layer=model."+FEATURE_LAYER)

#explainer = shap.GradientExplainer((model, features_layer), normalize(X), local_smoothing=0.5)
explainer = shap.GradientExplainer((model, features_layer), normalize(X), local_smoothing=0.5)
shap_values,indexes = explainer.shap_values(normalize(to_explain), ranked_outputs=RANKED_OUTPUTS, nsamples=IMG_SAMPLES)

# get the names for the classes
dic_class_names = {i :class_names[i] for i in range(0, len(class_names))}

index_names = np.vectorize(lambda x: dic_class_names[x])(indexes)

# plot the explanations
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

# image plot
def image_plot_v2(shap_values, x, labels=None, show=True, width=20, aspect=0.2, hspace=0.2, labelpad=None):

    input_image = list();
    curr_gray_image = list();
    multi_output = True
    
    if type(shap_values) != list:
        multi_output = False
        shap_values = [shap_values]
        
    # make sure labels
    if labels is not None:
        assert labels.shape[0] == shap_values[0].shape[0], "Labels must have same row count as shap_values arrays!"
        if multi_output:
            assert labels.shape[1] == len(shap_values), "Labels must have a column for each output in shap_values!"
        else:
            assert len(labels.shape) == 1, "Labels must be a vector for single output shap_values."
    
    label_kwargs = {} if labelpad is None else {'pad': labelpad}
    
    # plot our explanations
    fig_size = np.array([3 * (len(shap_values) + 1), 2.5 * (x.shape[0] + 1)])
       
    if fig_size[0] > width:
        fig_size *= width / fig_size[0]
    fig, axes = pl.subplots(nrows=x.shape[0], ncols=len(shap_values) + 1, figsize=fig_size)  
    
    if len(axes.shape) == 1:
        axes = axes.reshape(1,axes.size)
        
    for row in range(x.shape[0]):
        x_curr = x[row].copy()
    
        # make sure
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])
        
    
        if x_curr.max() > 1:
            x_curr /= 255.
    
        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (0.2989 * x_curr[:,:,0] + 0.5870 * x_curr[:,:,1] + 0.1140 * x_curr[:,:,2]) # rgb to gray
        else:
            x_curr_gray = x_curr
    
        axes[row,0].imshow(x_curr)
        axes[row,0].axis('off')
        
        input_image.append(x_curr)
        curr_gray_image.append(x_curr_gray)  
        
        
        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
        else:
            abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)
    
    
        for i in range(len(shap_values)):
            if labels is not None:
                axes[row,i+1].set_title(labels[row,i], **label_kwargs)
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
            axes[row,i+1].imshow(x_curr_gray, cmap=pl.get_cmap('gray'), alpha=0.15, extent=(-1, sv.shape[0], sv.shape[1], -1))
           
            
            colors = []
            for l in np.linspace(1, 0, 100):
                colors.append((30./255, 136./255, 229./255,l))
            for l in np.linspace(0, 1, 100):
                colors.append((255./255, 13./255, 87./255,l))
            
            cm= LinearSegmentedColormap.from_list("red_transparent_blue", colors)
            im = axes[row,i+1].imshow(sv, cmap=cm, vmin=-max_val, vmax=max_val)
            axes[row,i+1].axis('off')
    
    
    hspace = 0.5
    if hspace == 'auto':
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
        
    cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal", aspect=fig_size[0]/aspect)
    cb.outline.set_visible(False)

    fig.savefig('output.png')
    
    
##########
image_plot_v2(shap_values, to_explain, index_names)


def fig_to_base64(fig):
  img = io.BytesIO()
  pl.savefig(img)
  img.seek(0)

  return base64.b64encode(img.getvalue())

filename = "output.png"

with open(filename, mode='rb') as file: # b is important -> binary
  img = file.read()
  encoded = fig_to_base64(img)
  html_image = '<img class="img-fluid" src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
    
    
result = ''
with pd.option_context('display.max_colwidth', -1):
  result = html_image 

result = """     
            <!DOCTYPE html>
            <html>
              <head>
                <meta charset="UTF-8">
                  <title>Deep Learning Model Explainability</title>
                  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
                  </head>
                  <body class="container">
                    <h1 style="color:#003050;">
                      <center>Deep Learning Model Explainability</center>
                    </h1>
                    <br>
                      <br>
                        <br>
                          <h3 style="color:#003050;" id="M1">Explain a Deep learning Model using GradientExplainer</h3>
                          <p>Explaining a prediction in terms of the original input image is harder than explaining the predicition in terms of a higher convolutional layer (because the higher convolutional layer is closer to the output). Gradient explainer uses expected gradients, which merges ideas from integrated gradients, SHAP, and SmoothGrad into a single expection equation. Red pixels represent positive SHAP values that increase the probability of the class, while blue pixels represent negative SHAP values the reduce the probability of the class.</p>
                          <center>{0}</center>
                          <footer>
                            <hr/>
                            <ul class="nav justify-content-center">
                              <li class="nav-item">
                                <a class="nav-link active" href="https://www.activeeon.com/" target="_blank">
                                  <medium>Activeeon</medium>
                                </a>
                                <li class="nav-item">
                                  <a class="nav-link" href="MLOS/MLOSUserGuide.html" target="_blank">
                                    <medium>Machine Learning Open Studio</medium>
                                  </a>
                                </li>
                              </ul>
                            </footer>
                          </body></html>
""".format(html_image)

result = result.encode('utf-8')
resultMetadata.put("file.extension", ".html")
resultMetadata.put("file.name", "result.html")
resultMetadata.put("content.type", "text/html")

print("END Model_Explainability")