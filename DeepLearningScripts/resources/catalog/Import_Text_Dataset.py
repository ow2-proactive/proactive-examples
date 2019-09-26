print("BEGIN Import_Text_Dataset")

import os
import wget
import zipfile
import shutil
import random
import codecs
import numpy as np
import pandas as pd
from torchtext import data
#import spacy

from os import remove, listdir, makedirs
from os.path import basename, splitext, exists, join
from sklearn.model_selection import train_test_split
  
### PHASE 1 ################

DATASET_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/datasets/DL32.zip'
#DATASET_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/datasets/IMDB.zip'
#DATASET_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/datasets/unlabeled-IMDB.zip'
GLOBALSPACE = 'text_data/'
TRAIN_SPLIT = round(0.6, 3)
TEST_SPLIT  = round(0.3, 3)
VALIDATION_SPLIT = round(0.1, 3)
TOY_MODE = True
TOKENIZER = "spacy"
SENTENCE_SEPARATOR = '\r'
CHARSET = 'utf-8'
IS_LABELED_DATA = True
DATASET_ITERATOR_UNL = None

# READ TASK VARIABLES
if 'variables' in locals():
  
  if variables.get("DATASET_URL") is not None:
    DATASET_URL = variables.get("DATASET_URL")
  if variables.get("TRAIN_SPLIT") is not None:
    TRAIN_SPLIT = float(str(variables.get("TRAIN_SPLIT")))
  if variables.get("TEST_SPLIT") is not None:
    TEST_SPLIT = float(str(variables.get("TEST_SPLIT")))
  if variables.get("VALIDATION_SPLIT") is not None:
    VALIDATION_SPLIT = float(str(variables.get("VALIDATION_SPLIT")))
  if variables.get("TOY_MODE") is not None:
    TOY_MODE = variables.get("TOY_MODE")
  if variables.get("TOKENIZER") is not None:
    TOKENIZER = str(variables.get("TOKENIZER"))
  if variables.get("SENTENCE_SEPARATOR") is not None:
    SENTENCE_SEPARATOR = variables.get("SENTENCE_SEPARATOR")
  if variables.get("CHARSET") is not None:
    CHARSET = str(variables.get("CHARSET"))
  if variables.get("IS_LABELED_DATA") is not None:
    IS_LABELED_DATA = variables.get("IS_LABELED_DATA")

print("Split information:")
print("TRAIN_SPLIT:      " + str(TRAIN_SPLIT))
print("TEST_SPLIT:       " + str(TEST_SPLIT))
print("VALIDATION_SPLIT: " + str(VALIDATION_SPLIT))

assert TRAIN_SPLIT >= 0.0
assert TEST_SPLIT >= 0.0
assert VALIDATION_SPLIT >= 0.0
assert round(TRAIN_SPLIT + TEST_SPLIT + VALIDATION_SPLIT, 3) == 1
if TRAIN_SPLIT == 0.0 and VALIDATION_SPLIT > 0.0:
  raise AssertionError("VALIDATION_SPLIT cannot be defined when TRAIN_SPLIT equals zero") 

DATASET_PATH = os.path.join(GLOBALSPACE,splitext(DATASET_URL[DATASET_URL.rfind("/")+1:])[0])

if exists(DATASET_PATH):
  shutil.rmtree(DATASET_PATH)
makedirs(DATASET_PATH)

print("DATASET_URL:  " + DATASET_URL)
print("DATASET_PATH: " + DATASET_PATH)

# DOWNLOAD AND EXTRACT DATASET
print("Downloading...")
filename = wget.download(DATASET_URL, DATASET_PATH)
print("FILENAME: " + filename)
print("OK")

print("Extracting...")
dataset_zip = zipfile.ZipFile(filename)
dataset_zip.extractall(DATASET_PATH)
dataset_zip.close()
remove(filename)
print("OK")

### PHASE 2 ################

# EXTRACT LABELS
if IS_LABELED_DATA:
    textfolders = [os.path.join(root, name)
             for root, dirs, files in os.walk(DATASET_PATH)
             for name in dirs]

    labels = [os.path.join(name)
             for root, dirs, files in os.walk(DATASET_PATH)
             for name in dirs]
    print('labels to be predicted',labels)
    class_files = [os.path.join(root, name)
             for i in range(0,len(textfolders))
             for root, dirs, files in os.walk(textfolders[i])
             for name in files]
else:
    DATASET_PATH = os.path.join(DATASET_PATH,'unlabeled')
    labels = ['unlabeled']
    class_files = [os.path.join(DATASET_PATH, name)
            for root, dirs, files in os.walk(DATASET_PATH)
            for name in files]
    #assert(len(class_files)==0)



### PHASE 3 ################

### SPLIT DATASET
import codecs
import random
import pandas as pd

sent_classes={}
n_class=0
toy_dataset_size = 2000


train_data = []
val_data = []
test_data = []

for i in range(len(class_files)):
    if class_files[i].endswith('.DS_Store'):
        continue
    print('loading MR data from',class_files[i])
    sent_classes[labels[n_class]] = codecs.open(class_files[i], 'r', CHARSET).read().strip().splitlines()
    print('length of class',len(sent_classes[labels[n_class]]))
    random.shuffle(sent_classes[labels[n_class]])
    file_len = len(sent_classes[labels[n_class]])
    if TOY_MODE:
        class_ent_len = int(toy_dataset_size/len(labels))
        if (file_len<class_ent_len):
            class_ent_len = file_len
    else:
        class_ent_len = file_len
    
    train_data = train_data + [(sent,labels[n_class]) for sent in sent_classes[labels[n_class]][:int(class_ent_len*TRAIN_SPLIT)]]
    val_data = val_data + [(sent,labels[n_class]) for sent in sent_classes[labels[n_class]][int(class_ent_len*TRAIN_SPLIT+1):int(class_ent_len*(TRAIN_SPLIT+VALIDATION_SPLIT)+1)]]
    test_data = test_data + [(sent,labels[n_class]) for sent in sent_classes[labels[n_class]][int(class_ent_len*(TRAIN_SPLIT+VALIDATION_SPLIT)+2):class_ent_len]]
    n_class = n_class+1

train_frame = pd.DataFrame(train_data, columns = ["text","label"])
val_frame = pd.DataFrame(val_data, columns = ["text","label"])
test_frame = pd.DataFrame(test_data, columns = ["text","label"])

train_frame['text'].replace('', np.nan, inplace=True)
train_frame.dropna(subset=['text'], inplace=True)
val_frame['text'].replace('', np.nan, inplace=True)
val_frame.dropna(subset=['text'], inplace=True)
test_frame['text'].replace('', np.nan, inplace=True)
test_frame.dropna(subset=['text'], inplace=True)

train_path = os.path.join(DATASET_PATH,"train.csv")
val_path = os.path.join(DATASET_PATH,"val.csv")
test_path = os.path.join(DATASET_PATH,"test.csv")

import os
arr = os.listdir()
print("list_folders",arr)
#arr = os.listdir("text_data/IMDB/unlabeled")
#print("list_folders_text_data",arr)

train_frame.to_csv(train_path, encoding=CHARSET,index=False, header=False)
val_frame.to_csv(val_path, encoding=CHARSET, index=False, header=False)
test_frame.to_csv(test_path, encoding=CHARSET, index=False, header=False)
print(train_path)

### PHASE 4 ###################

DATASET_ITERATOR="""
text_field = data.Field(lower=True)#, tokenize=TOKENIZER)
label_field = data.Field(sequential=False)
#Dataset of columns stored in CSV, TSV, or JSON format
train, val, test = data.TabularDataset.splits(path=DATASET_PATH, train='train.csv',
                                                  validation='val.csv', test='test.csv', format='csv',
                                                  fields=[('text', text_field), ('label', label_field)])
train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test),
                                                              repeat=False,
                                                             batch_sizes=(len(train),len(val),len(test)), sort_key=lambda x: len(x.text), device=DEVICE)
if variables.get(DATASET_ITERATOR_UNL) is None:
    text_field.build_vocab(train)
    label_field.build_vocab(train)
    VOCAB_SIZE=len(text_field.vocab)
    LABEL_SIZE=len(label_field.vocab)
"""
if 'variables' in locals():
    if IS_LABELED_DATA:
        variables.put("DATASET_ITERATOR",DATASET_ITERATOR)
        variables.put("DATASET_PATH",DATASET_PATH)
    else:
        variables.put("DATASET_ITERATOR_UNL",DATASET_ITERATOR)
        variables.put("DATASET_PATH_UNL",DATASET_PATH)
    variables.put("TOKENIZER",TOKENIZER)
    variables.put("IS_LABELED_DATA",IS_LABELED_DATA)
    
print("END Import_Text_Dataset")