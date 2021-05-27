
print("BEGIN Predict_Text_Classification_Model")

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import dill as pickle
from itertools import *
import time, random
from tqdm import tqdm
from torchtext import data
from torchtext import vocab
import torch.optim as optim
from torchtext import datasets
import torch.nn.functional as F
from torch.autograd import Variable
import xml.sax.saxutils as saxutils 
from torchtext.vocab import Vectors, FastText, GloVe, CharNGram

pd.options.display.max_colwidth = 500
DEVICE = -1
#-------------------------main code --------------------------
DATASET_ITERATOR_UNL = None
DATASET_ITERATOR = None
#--------Get varaiables from previous tasks--------
if 'variables' in locals():
    if variables.get("MODEL_PATH") is not None:
        MODEL_PATH = variables.get("MODEL_PATH")
    if variables.get("LOSS_FUNCTION") is not None:
        LOSS_FUNCTION = variables.get("LOSS_FUNCTION")
    if variables.get("BATCH_SIZE") is not None:
        BATCH_SIZE = variables.get("BATCH_SIZE")
    if variables.get("USE_GPU") is not None:
        USE_GPU = variables.get("USE_GPU")
    if variables.get("DEVICE") is not None:
        DEVICE = variables.get("DEVICE") 
    if variables.get("IS_LABELED_DATA") is not None:
        IS_LABELED_DATA = variables.get("IS_LABELED_DATA")
    if variables.get("vocab_size") is not None:
        vocab_size = variables.get("vocab_size")
    if variables.get("label_size") is not None:
        label_size = variables.get("label_size")
    if variables.get("BATCH_SIZE") is not None:
        label_size = variables.get("BATCH_SIZE")
    if variables.get("MODEL_CLASS") is not None:
        MODEL_CLASS = variables.get("MODEL_CLASS")
    if variables.get("MODEL_DEF") is not None:
        MODEL_DEF = variables.get("MODEL_DEF")


#--------Load Dataset--------

if 'variables' in locals():
    if variables.get("DATASET_PATH") is not None:
        DATASET_PATH = variables.get("DATASET_PATH")
    if variables.get("DATASET_ITERATOR") is not None:
        DATASET_ITERATOR = variables.get("DATASET_ITERATOR")
        DATASET_PATH = variables.get("DATASET_PATH")
        exec(DATASET_ITERATOR)
    if variables.get("DATASET_ITERATOR_UNL") is not None:
        DATASET_ITERATOR_UNL = variables.get("DATASET_ITERATOR_UNL")
        DATASET_PATH = variables.get("DATASET_PATH")
        exec(DATASET_ITERATOR_UNL)
    #Load model files
    if variables.get("LABELS_PATH") is not None:
        LABELS_PATH = variables.get("LABELS_PATH")
    if variables.get("TEXT_PATH") is not None:
        TEXT_PATH = variables.get("TEXT_PATH")
   
#-------Main--------
def evaluate(model, test, data_iter, label_field, loss_function, name):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    i=0
    acc = 0
    pd.options.display.max_colwidth = 500
    result =  pd.DataFrame(columns=['text','Predictions','Targets'])
    for batch in data_iter:
        i=i+1
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(test)
        model.hidden = model.init_hidden()
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        for i in range(model.batch_size):
            test_fields = vars(test[i])
            test_text = test_fields["text"]
            test_label = test_fields["label"]
            prede_label = pred_label[i]
            gd_label = label[i]
            result.loc[i] = [' '.join(test_text),label_field.vocab.itos[prede_label+1], test_label]
        pred_res += [x for x in pred_label]
        if name is 'test_labeled':
            loss = loss_function(pred, label)
            avg_loss += loss.item()
    if name is 'test_labeled':
        avg_loss /= len(test)
        pred_res = torch.LongTensor(pred_res)
        acc = get_accuracy(truth_res, pred_res)
        print(name + ': loss %.2f acc %.1f' % (avg_loss, acc*100))
    return acc, result
    
def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)
     
LOSS ="""loss_function = nn."""+LOSS_FUNCTION+"""()"""

exec(LOSS)


label_field = pickle.load(open(LABELS_PATH,'rb'))
text_field = pickle.load(open(TEXT_PATH,'rb'))
exec(MODEL_CLASS)

MODEL=torch.load(MODEL_PATH)
if variables.get("DATASET_ITERATOR_UNL") is not None:
    print('I am testing the unlabeled dataset')
    test_acc, results = evaluate(MODEL, test, test_iter, label_field, loss_function, 'test_unlabeled')
else:
    print('I am testing the labeled dataset')
    test_acc, results = evaluate(MODEL, test, test_iter, label_field, loss_function, 'test_labeled')

reponse_good = '&#9989;'
reponse_bad = '&#10060;'
results['Results'] = np.where((results['Predictions'] == results['Targets']), saxutils.unescape(reponse_good), saxutils.unescape(reponse_bad)) 

#------plot results-----
# Forward results for preview 
try:
    variables.put("PREDICT_DATA_JSON", results.to_json(orient='split'))
except NameError as err:
    print("{0}".format(err))
    print("Warning: this script is running outside from ProActive.")
    pass

print("END Predict_Text_Classification_Model")