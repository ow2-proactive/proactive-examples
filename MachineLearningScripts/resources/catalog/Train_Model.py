__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import sys, bz2, uuid, json
import random, pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

is_labeled_data = False
LABEL_COLUMN = variables.get("LABEL_COLUMN")
if LABEL_COLUMN is not None and LABEL_COLUMN is not "":
  is_labeled_data = True

input_variables = {
  'task.dataframe_id': None, 
  'task.dataframe_id_train': None,
  'task.algorithm_json': None
}

for key in input_variables.keys():
  for res in results:
    value = res.getMetadata().get(key)
    if value is not None:
      input_variables[key] = value
      break

dataframe_id = None
if input_variables['task.dataframe_id'] is not None:
  dataframe_id = input_variables['task.dataframe_id']
if input_variables['task.dataframe_id_train'] is not None:
  dataframe_id = input_variables['task.dataframe_id_train']
print("dataframe id (in): ", dataframe_id)

dataframe_json = variables.get(dataframe_id)
assert dataframe_json is not None
dataframe_json = bz2.decompress(dataframe_json).decode()
dataframe = pd.read_json(dataframe_json, orient='split')

algorithm_json = input_variables['task.algorithm_json']
assert algorithm_json is not None
algorithm = json.loads(algorithm_json)

#-------------------------------------------------------------
class obj(object):
  def __init__(self, d):
    for a, b in d.items():
      if isinstance(b, (list, tuple)):
        setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
      else:
        setattr(self, a, obj(b) if isinstance(b, dict) else b)
#-------------------------------------------------------------
alg = obj(algorithm)
loss = 0
try:
    vars = json.loads(alg.input_variables)
except:
    vars = None
try:
    automl = alg.automl
except:
    automl = True
try:
    if alg.sampling:
        print(alg.sampling + "exists")
except:
    alg.sampling=False
model = None
print("alg.is_supervised",alg.is_supervised)
print("alg.name",alg.name)
print("alg.type",alg.type)

if alg.is_supervised:
  #-------------------------------------------------------------
  # Classification algorithms
  #
  if alg.name == 'TPOT_Classifier':
    from tpot import TPOTClassifier
    model = TPOTClassifier(        
        generations=alg.generations,
        cv=alg.cv,
        scoring=alg.scoring,
        verbosity=alg.verbosity)
  elif alg.name == 'AutoSklearn_Classifier':
    from autosklearn import classification
    if alg.sampling.lower()=='true':
      model = classification.AutoSklearnClassifier(
          time_left_for_this_task=alg.task_time,
          per_run_time_limit=alg.run_time,
          resampling_strategy= "".join(alg.sampling_strategy),
          resampling_strategy_arguments={'folds':int(alg.folds)}
          #feat_type = {Numerical,Numerical,Numerical,Numerical,Categorical}
      )
    else:
        model = classification.AutoSklearnClassifier(
          time_left_for_this_task=alg.task_time,
          per_run_time_limit=alg.run_time
      )
  elif alg.name == 'SupportVectorMachines':
    from sklearn.svm import SVC
    model = SVC(**vars)   
  elif alg.name == 'GaussianNaiveBayes':
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB(**vars)  
  elif alg.name == 'LogisticRegression':
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(**vars)

  #-------------------------------------------------------------
  # Regression algorithms   
  elif alg.name == 'TPOT_Regressor':
    from tpot import TPOTRegressor
    model = TPOTRegressor(
        generations=alg.generations,
        cv=alg.cv,
        scoring=alg.scoring,
        verbosity=alg.verbosity
    )
  elif alg.name == 'AutoSklearn_Regressor':
    from autosklearn import regression
    print("alg.sampling",alg.sampling_strategy)
    if alg.sampling.lower()=='true':
      model = regression.AutoSklearnRegressor(
          time_left_for_this_task=alg.task_time,
          per_run_time_limit=alg.run_time,
          resampling_strategy= "".join(alg.sampling_strategy),
          resampling_strategy_arguments={'folds':int(alg.folds)}
          #feat_type = {Numerical,Numerical,Numerical,Numerical,Categorical}
      )
    else:
        model = regression.AutoSklearnRegressor(
          time_left_for_this_task=alg.task_time,
          per_run_time_limit=alg.run_time
      )
  elif alg.name == 'LinearRegression':
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(**vars)
  elif alg.name == 'SupportVectorRegression':
    from sklearn.svm import SVR
    model = SVR(**vars)
  elif alg.name == 'BayesianRidgeRegression':
    from sklearn.linear_model import BayesianRidge
    model = BayesianRidge(**vars)
else:
  #-------------------------------------------------------------
  # Anomaly detection algorithms
  if alg.name == 'OneClassSVM':
    from sklearn import svm
    model = svm.OneClassSVM(**vars) 
  elif alg.name == 'IsolationForest':
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(**vars)
  #-------------------------------------------------------------
  # Clustering algorithms
  elif alg.name == 'MeanShift':
    from sklearn.cluster import MeanShift
    model = MeanShift(**vars)  
  elif alg.name == 'KMeans':
    from sklearn.cluster import KMeans
    model = KMeans(**vars)

#-------------------------------------------------------------
if model is not None:
  if is_labeled_data:
    columns = [LABEL_COLUMN]
    dataframe_train = dataframe.drop(columns, axis=1, inplace=False)
    dataframe_label = dataframe.filter(columns, axis=1)
  else:
    dataframe_train = dataframe
    
  if alg.is_supervised:
    print(dataframe_train.head())
    print(dataframe_label.head())
    model.fit(dataframe_train.values, dataframe_label.values.ravel())
    if (alg.type == 'classification' or alg.type == 'anomaly') and automl:
      scores = cross_val_score(model, dataframe_train.values, dataframe_label.values.ravel(), cv=int(variables.get("N_SPLITS")), scoring=alg.scoring)
      loss = 1 - np.mean(scores)
    if alg.type == 'regression' and automl:
      scores = cross_val_score(model, dataframe_train.values, dataframe_label.values.ravel(), cv=int(variables.get("N_SPLITS")), scoring=alg.scoring)
      loss = 1 - np.mean(scores)
    if alg.sampling:
      model.refit(dataframe_train.values.copy(), dataframe_label.values.ravel().copy())
  else:
    model.fit(dataframe_train.values)
    if is_labeled_data and automl:
        scores = cross_val_score(model, dataframe_train.values, dataframe_label.values.ravel(), cv=int(variables.get("N_SPLITS")), scoring=alg.scoring)
        loss = 1 - np.mean(scores)
  if alg.name == 'TPOT_Regressor' or alg.name =='TPOT_Classifier':
    model = model.fitted_pipeline_
  model_bin = pickle.dumps(model)
  model_compressed = bz2.compress(model_bin)
  model_id = str(uuid.uuid4())
  variables.put(model_id, model_compressed)

  print("model id: ", model_id)
  print('model size (original):   ', sys.getsizeof(model_bin), " bytes")
  print('model size (compressed): ', sys.getsizeof(model_compressed), " bytes")   
  resultMetadata.put("task.model_id", model_id)
else:
  print("Algorithm not found!")

dataframe_json = dataframe.to_json(orient='split').encode()
compressed_data = bz2.compress(dataframe_json)

dataframe_id = str(uuid.uuid4())
variables.put(dataframe_id, compressed_data)

print("dataframe id (out): ", dataframe_id)
print('dataframe size (original):   ', sys.getsizeof(dataframe_json), " bytes")
print('dataframe size (compressed): ', sys.getsizeof(compressed_data), " bytes")
print(dataframe.head())

resultMetadata.put("task.name", __file__)
#resultMetadata.put("task.dataframe_id", dataframe_id)
resultMetadata.put("task.algorithm_json", algorithm_json)
resultMetadata.put("task.label_column", LABEL_COLUMN)

token = variables.get("TOKEN")
# Convert from JSON to dict
token = json.loads(token)

# return the loss value
result_map = {
    'token': token,
    'loss': loss
}

result_map = json.dumps(result_map)
resultMap.put("RESULT_JSON", result_map)
print('result_map: ', result_map)

print("END " + __file__)