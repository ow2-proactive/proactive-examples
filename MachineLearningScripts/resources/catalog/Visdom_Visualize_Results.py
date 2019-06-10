__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() != 'true':
  print("Task " + __file__ + " disabled")
  quit()

print("BEGIN " + __file__)

import sys, bz2, uuid, json
import pandas as pd
import numpy as np

from visdom import Visdom
from sklearn.metrics import *
from pandas.api.types import is_string_dtype

input_variables = {
  'task.dataframe_id': None,
  'task.dataframe_id_test': None,
  'task.algorithm_json': None,
  'task.label_column': None,
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
if input_variables['task.dataframe_id_test'] is not None:
  dataframe_id = input_variables['task.dataframe_id_test']
print("dataframe id (in): ", dataframe_id)

dataframe_json = variables.get(dataframe_id)
assert dataframe_json is not None
dataframe_json = bz2.decompress(dataframe_json).decode()

dataframe = pd.read_json(dataframe_json, orient='split')

is_labeled_data = False
LABEL_COLUMN = variables.get("LABEL_COLUMN")
if LABEL_COLUMN is not None and LABEL_COLUMN is not "":
  is_labeled_data = True
else:
  LABEL_COLUMN = input_variables['task.label_column']
  if LABEL_COLUMN is not None and LABEL_COLUMN is not "":
    is_labeled_data = True

TARGET_CLASS = variables.get("TARGET_CLASS")
assert TARGET_CLASS is not None, 'The variable TARGET_CLASS is mandatory'
TARGET_CLASS = str(TARGET_CLASS)

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

visdom_endpoint = variables.get("VISDOM_ENDPOINT") if variables.get("VISDOM_ENDPOINT") else results[0].__str__()
print("VISDOM_ENDPOINT = ",visdom_endpoint)
if visdom_endpoint is not None:
  visdom_endpoint = visdom_endpoint.replace("http://", "")
(VISDOM_HOST, VISDOM_PORT) = visdom_endpoint.split(":")
print("Connecting to %s:%s" % (VISDOM_HOST, VISDOM_PORT))
vis = Visdom(server="http://"+VISDOM_HOST,port=int(VISDOM_PORT))
assert vis.check_connection()

columns = []
if alg.type == 'classification' and is_labeled_data:
  columns = [LABEL_COLUMN, "predictions", "results"]
elif alg.type == 'regression' and is_labeled_data:
  columns = [LABEL_COLUMN, "predictions", "absolute_error"]
elif alg.type == 'clustering' and is_labeled_data:
  columns = [LABEL_COLUMN, "predictions"]
else:
  columns = ["predictions"]

dataframe_idx_values = dataframe.index.values
dataframe_features = dataframe.drop(columns, axis=1, inplace=False)

dataframe_features_values = dataframe_features.values
dataframe_features_values_columns = list(dataframe_features.columns.values)

nb_rows = dataframe_features_values.shape[0]
nb_columns = dataframe_features_values.shape[1]

dataframe_predictions = dataframe["predictions"]
dataframe_predictions_values = dataframe_predictions.values.ravel()
dataframe_predictions_values_str = [str(dataframe_predictions[x]) for x in range(nb_rows)]

if alg.type == 'classification':
  df_classes = dataframe[LABEL_COLUMN].unique()
else:
  df_classes = dataframe_predictions.unique()

classes = [str(df_classes[x]) for x in range(0,len(df_classes))]
classes.sort()
classes_stats = [0]*len(classes)

if is_labeled_data:
  dataframe_labels = dataframe[LABEL_COLUMN]
  dataframe_labels_values = dataframe_labels.values.ravel()

if is_labeled_data and alg.type == 'regression':
  dataframe_predictions_values_float = [float(dataframe_predictions_values[x]) for x in range(nb_rows)]
  model_mse = mean_squared_error(dataframe_labels_values, dataframe_predictions_values_float)
  model_mae = mean_absolute_error(dataframe_labels_values, dataframe_predictions_values_float)
  model_r2s = r2_score(dataframe_labels_values, dataframe_predictions_values_float)

if is_labeled_data and (alg.type == 'classification' or alg.type == 'clustering'):
  dataframe_classes = dataframe_labels.unique()
  classes = [str(dataframe_classes[x]) for x in range(0, len(dataframe_classes))]
  classes.sort()
  classes_stats = [0]*len(classes)
  model_cm = confusion_matrix(dataframe_labels_values, dataframe_predictions_values)
  model_as = accuracy_score(dataframe_labels_values, dataframe_predictions_values)
  model_ps = precision_score(dataframe_labels_values, dataframe_predictions_values, average='micro')
  model_fpr = None
  model_tpr = None
  if len(classes) == 2 and not is_string_dtype(dataframe[LABEL_COLUMN]):
    model_fpr, model_tpr, _ = roc_curve(dataframe_labels_values, dataframe_predictions_values)

#-------------------------------------------------------------
# VISDOM PLOTS
#-------------------------------------------------------------

list_detected_samples_text = vis.text("List of detected samples in class " + TARGET_CLASS, opts=dict(title='List of indexes'))

target_class_line = vis.line(
  Y=np.array([0]),
  X=np.array([0]),
  opts=dict(
    xlabel='Index',
    ylabel='Targeted Class',
    title="Detected samples in class " + TARGET_CLASS
    )
  )

features_line = vis.line(
  X=np.column_stack([0]*nb_columns),
  Y=np.column_stack(dataframe_features_values[0][:]),
  opts=dict(
    legend=dataframe_features_values_columns,
    xlabel='Index',
    ylabel='Feature Value',
    title='Values of extracted features for each sample'
    )
  )
if alg.type == 'classification' or alg.type == 'clustering':
  statistic_pie = vis.pie(X=classes_stats, opts=dict(legend=classes, title='Classification results'))

#-------------------------------------------------------------
count = 0
for x in range(nb_rows):
  vis.line(
    X=np.column_stack([count]*nb_columns),
    Y=np.column_stack(dataframe_features_values[x][:]),
    win=features_line,
    update='append'
  )
  if alg.type == 'regression':
    match = (int(float(dataframe_predictions_values[x])) == int(float(TARGET_CLASS)))
    if match:
      message = "%s\n"%(dataframe_idx_values[x])
      vis.text(message, win=list_detected_samples_text, append=True)
      vis.line(Y=np.array([1]), X=np.array([count]), win=target_class_line, update='append')
    else:
      vis.line(Y=np.array([0]), X=np.array([count]), win=target_class_line, update='append')
    count += 1

  if alg.type == 'classification' or alg.type == 'clustering':
    for i in range(len(classes)):
      if dataframe_predictions_values_str[x] == classes[i]:
        classes_stats[i] += 1
        vis.pie(X=classes_stats, win=statistic_pie, opts=dict(legend=classes, title='Classification results'))
        match = (str(classes[i]) == str(TARGET_CLASS))
        if match:
          message = "%s\n"%(dataframe_idx_values[x])
          vis.text(message, win=list_detected_samples_text, append=True)
          vis.line(Y=np.array([1]), X=np.array([count]), win=target_class_line, update='append')
        else:
          vis.line(Y=np.array([0]), X=np.array([count]), win=target_class_line, update='append')
        count += 1

#-------------------------------------------------------------

score_text = vis.text("Model scoring")
if is_labeled_data and (alg.type == 'classification' or alg.type == 'clustering'):
  if model_fpr is not None and model_tpr is not None:
    vis.line(X=model_fpr, Y=model_tpr, opts=dict(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve'))
  vis.bar(X=model_cm, opts=dict(stacked=True, legend=classes, rownames=classes, title='Predictive model performance'))
  vis.text("Classification scores", win=score_text, append=True)
  vis.text("Accuracy score:", win=score_text, append=True)
  vis.text(str(model_as), win=score_text, append=True)
  vis.text("Precision score:", win=score_text, append=True)
  vis.text(str(model_ps), win=score_text, append=True)
  vis.text("Confusion matrix:", win=score_text, append=True)
  vis.text(str(model_cm), win=score_text, append=True)

if is_labeled_data and alg.type == 'regression':
  vis.text("Regression scores", win=score_text, append=True)
  vis.text("Mean squared error:", win=score_text, append=True)
  vis.text(str(model_mse), win=score_text, append=True)
  vis.text("Mean absolute error:", win=score_text, append=True)
  vis.text(str(model_mae), win=score_text, append=True)
  vis.text("Coefficient of determination:", win=score_text, append=True)
  vis.text(str(model_r2s), win=score_text, append=True)

print("END " + __file__)
