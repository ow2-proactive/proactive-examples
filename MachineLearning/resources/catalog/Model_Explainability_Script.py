
# -*- coding: utf-8 -*-
"""Proactive Model Explainability for Machine Learning

This module contains the Python script for the Model Explainability task.
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import ssl
import urllib.request
import io
import eli5
import base64
import string
import pandas as pd
import sys, bz2
import random, pickle
import shap

from eli5.sklearn import PermutationImportance
from pdpbox import pdp

global variables, resultMetadata

__file__ = variables.get("PA_TASK_NAME")
print("BEGIN " + __file__)

# -------------------------------------------------------------
# Get schedulerapi access and acquire session id
schedulerapi.connect()
sessionid = schedulerapi.getSession()

# -------------------------------------------------------------
# Import an external python script containing a collection of
# common utility Python functions and classes
PA_CATALOG_REST_URL = variables.get("PA_CATALOG_REST_URL")
PA_PYTHON_UTILS_URL = PA_CATALOG_REST_URL + "/buckets/machine-learning/resources/Utils_Script/raw"
req = urllib.request.Request(PA_PYTHON_UTILS_URL)
req.add_header('sessionid', sessionid)
if PA_PYTHON_UTILS_URL.startswith('https'):
    content = urllib.request.urlopen(req, context=ssl._create_unverified_context()).read()
else:
    content = urllib.request.urlopen(req).read()
exec(content, globals())
global check_task_is_enabled, get_input_variables, get_and_decompress_dataframe


def fig_to_base64(fig):
    img = io.BytesIO()
    plt.savefig(img)
    img.seek(0)
    return base64.b64encode(img.getvalue())


# -------------------------------------------------------------
# Check if the Python task is enabled or not
check_task_is_enabled()

# -------------------------------------------------------------
# Get data from the propagated variables
#
input_variables = {
    'task.dataframe_id': None,
    'task.dataframe_id_test': None,
    'task.label_column': None,
    'task.model_id': None
}
get_input_variables(input_variables)

dataframe_id = None
if input_variables['task.dataframe_id'] is not None:
    dataframe_id = input_variables['task.dataframe_id']
if input_variables['task.dataframe_id_test'] is not None:
    dataframe_id = input_variables['task.dataframe_id_test']
print("dataframe id (in): ", dataframe_id)

dataframe = get_and_decompress_dataframe(dataframe_id)

is_labeled_data = False
LABEL_COLUMN = variables.get("LABEL_COLUMN")
if LABEL_COLUMN is not None and LABEL_COLUMN is not "":
    is_labeled_data = True
else:
    LABEL_COLUMN = input_variables['task.label_column']
    if LABEL_COLUMN is not None and LABEL_COLUMN is not "":
        is_labeled_data = True

model_id = input_variables['task.model_id']
model_compressed = variables.get(model_id)
model_bin = bz2.decompress(model_compressed)
assert model_bin is not None
print("model id (in): ", model_id)
print("model size: ", sys.getsizeof(model_compressed), " bytes")
print("model size (decompressed): ", sys.getsizeof(model_bin), " bytes")

loaded_model = pickle.loads(model_bin)

if is_labeled_data:
    feature_partial = variables.get("FEATURE_PARTIAL_PLOTS")
    feature_partial_plots = [x.strip() for x in feature_partial.split(',')]
    features_to_plot = variables.get("FEATURE_PARTIAL2D_PLOTS")
    features_to_plot2d = [x.strip() for x in features_to_plot.split(',')]
    shap_row_to_show = int(variables.get("SHAP_ROW_SHOW"))
    columns = [LABEL_COLUMN]
    dataframe_test = dataframe.drop(columns, axis=1, inplace=False)

    dataframe_label = dataframe.filter(columns, axis=1)
    feature_names = dataframe_test.columns.values

    # -------------------------------------------------------------
    # PERMUTATION IMPORTANCE
    perm = PermutationImportance(loaded_model, random_state=1).fit(dataframe_test.values,
                                                                   dataframe_label.values.ravel())
    html_table = eli5.show_weights(perm, feature_names=dataframe_test.columns.tolist(), top=50)

    # -------------------------------------------------------------
    # PARTIAL DEPENDENCE PLOTS
    partial_feature_find = [i for i in feature_partial_plots if i in feature_names]
    html_partial_plot = ''
    for i in partial_feature_find:
        pdp_feature = pdp.pdp_isolate(model=loaded_model, dataset=dataframe_test,
                                      model_features=feature_names, feature=i)
        pdp_plot_feature = pdp.pdp_plot(pdp_feature, i)
        graph_name = ''.join(random.sample((string.ascii_uppercase + string.digits), 3))
        html_pdp = 'html_pdp_plot' + graph_name + ' + '
        encoded = fig_to_base64(pdp_plot_feature)
        html_pdp = '<img class="img-fluid" src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
        html_partial_plot += html_pdp

    # -------------------------------------------------------------
    # 2D PARTIAL DEPENDENCE PLOTS
    # Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate
    # and pdp_interact_plot instead of pdp_isolate_plot
    # features_to_plot = ['preg', 'skin']
    inter1 = pdp.pdp_interact(model=loaded_model, dataset=dataframe_test,
                              model_features=feature_names, features=features_to_plot2d)
    partial_plot = pdp.pdp_interact_plot(pdp_interact_out=inter1,
                                         feature_names=features_to_plot2d)  # plot_type='contour'  plot_type='grid'
    encoded = fig_to_base64(partial_plot)
    html_partial_plot2d = '<img class="img-fluid" src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))

    # -------------------------------------------------------------
    # SHAP PLOT
    data_for_prediction = dataframe_test.iloc[shap_row_to_show]
    explainer = shap.KernelExplainer(loaded_model.predict_proba, dataframe_test.values)
    shap_values = explainer.shap_values(data_for_prediction)
    shap.initjs()
    shap_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction,
                                matplotlib=True, show=False)
    encoded3 = fig_to_base64(shap_plot)
    html_shap_plot = '<img class="img-fluid" src="data:image/png;base64, {}">'.format(encoded3.decode('utf-8'))

result = ''
with pd.option_context('display.max_colwidth', -1):
    result = html_table.data + html_shap_plot + html_partial_plot + html_partial_plot2d

result = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Machine Learning Model Explainability</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" 
integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
</head>
<body class="container">
<h1 style="color:#003050;">
<center>Model Explainability</center>
</h1>
<ul class="nav justify-content-center">
<li class="nav-item">
<a class="nav-link active" href="#M1">Permutation Importance</a>
</li>
<li class="nav-item">
<a class="nav-link" href="#M2">Partial Dependence Plots</a>
</li>
<li class="nav-item">
<a class="nav-link" href="#M3">2D Partial Dependence Plots</a>
</li>
<li class="nav-item">
<a class="nav-link" href="#M4">SHAP Values</a>
</li>
</ul>
<br/><br/><br/>
<h3 style="color:#003050;" id="M1">Permutation Importance</h3>
<p>The values towards the top are the most important features, and those towards the bottom matter least. 
The first number in each row shows how much model performance decreased with a random shuffling 
(in this case, using "accuracy" as the performance metric). 
The number after the ± measures how performance varied from one-reshuffling to the next 
(after repeating the shuffling of the columns different times).</p>  
{0}
<hr/>
<h3 style="color:#003050;" id="M2">Partial Dependence Plots</h3>
<p>While feature importance shows what variables most affect predictions, 
partial dependence plots show how a feature affects predictions. 
The y axis is interpreted as change in the prediction from what it would be predicted at the baseline 
or leftmost value. A blue shaded area indicates level of confidence.</p>
{1}
<hr/>
<h3 style="color:#003050;" id="M3">2D Partial Dependence Plots</h3>
<p>It shows predictions for any combination of the features.</p>
{2}
<hr/>
<h3 style="color:#003050;" id="M4">SHAP Values</h3>
<p> SHAP values interpret the impact of having a certain value for a given feature in comparison 
to the prediction we'd make if that feature took some baseline value. 
We will look at SHAP values for a single row of the dataset (we chose row 1). 
Feature values causing increased predictions are in pink, 
and their visual size shows the magnitude of the feature's effect. 
Feature values decreasing the prediction are in blue.</p> 
{3}
<footer>
<hr/>
<ul class="nav justify-content-center">
<li class="nav-item">
<a class="nav-link active" href="https://www.activeeon.com/" target="_blank">
<medium>Activeeon</medium>
</a>
<li class="nav-item">
<a class="nav-link" href="PML/PMLUserGuide.html" target="_blank">
<medium>Machine Learning Open Studio</medium>
</a>
</li>
</ul>
</footer>
</body></html>
""".format(html_table.data, html_partial_plot, html_partial_plot2d, html_shap_plot)

# result = result.encoded.decode('utf-8')
result = result.encode('utf-8')
resultMetadata.put("file.extension", ".html")
resultMetadata.put("file.name", "result.html")
resultMetadata.put("content.type", "text/html")

print("END " + __file__)