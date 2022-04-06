
# -*- coding: utf-8 -*-
"""Proactive Import Data for Machine Learning

This module contains the Python script for the Import Data and Automate Feature Engineering task.
"""
import os
import ssl
import uuid
import ast
import math
import time
import pickle
import socket
import urllib.request
import numpy as np
import pandas as pd
import yaml
import category_encoders as ce

from io import StringIO
from threading import Thread
from numpy import sort, array
from flask import Flask, render_template, Response, request
from tornado.ioloop import IOLoop
from sklearn.preprocessing import scale, LabelEncoder
from tensorflow.keras.models import load_model

from bokeh.embed import server_document
from bokeh.layouts import row, column, grid
from bokeh.models import ColumnDataSource, TableColumn, DataTable, CustomJS
from bokeh.models import Panel, Tabs
from bokeh.models import RadioGroup, RadioButtonGroup, NumericInput
from bokeh.models import Select, Div, Button
from bokeh.models.widgets import TextInput
from bokeh.server.server import Server
from bokeh.themes import Theme

pd.options.mode.chained_assignment = None  # default='warn'

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
global check_task_is_enabled, preview_dataframe_in_task_result
global import_csv_file, compress_and_transfer_dataframe
global assert_not_none_not_empty, str_to_bool

# -------------------------------------------------------------
# Check if the Python task is enabled or not
check_task_is_enabled()

# -------------------------------------------------------------
# Get data from the propagated variables
#
IMPORT_FROM = variables.get("IMPORT_FROM")
FILE_PATH = variables.get("FILE_PATH")
FILE_DELIMITER = variables.get("FILE_DELIMITER")

assert_not_none_not_empty(IMPORT_FROM, "IMPORT_FROM should be defined!")
assert_not_none_not_empty(FILE_PATH, "FILE_PATH should be defined!")
assert_not_none_not_empty(FILE_DELIMITER, "FILE_DELIMITER should be defined!")

# -------------------------------------------------------------
# Load file
#
dataframe = import_csv_file(FILE_PATH, FILE_DELIMITER, IMPORT_FROM)
feature_names = dataframe.columns

# -------------------------------------------------------------
# Create a local copy of the dataframe
#
ID = str(uuid.uuid4())
file_path = os.path.join('.', ID + '.csv')
dataframe.to_csv(file_path, sep=FILE_DELIMITER, index=False, header=True)
file_path_new = os.path.join('.', ID + '.new.csv')
file_path_params = os.path.join('.', ID + '.params.csv')
open(file_path_new, 'a').close()
open(file_path_params, 'a').close()

# -------------------------------------------------------------
# Configure network settings
#
def get_open_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("",0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port
hostname = socket.gethostname()
# local_ip = socket.gethostbyname(hostname)
external_ip = urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
bokeh_port = get_open_port() # 5007
flask_port = get_open_port() # 8001
bokeh_url = "http://"+external_ip+":"+str(bokeh_port)
flask_url = "http://"+external_ip+":"+str(flask_port)

schedulerapi.addExternalEndpointUrl(variables.get("PA_JOB_ID"), "AutoFeat", flask_url, "/automation-dashboard/styles/patterns/img/wf-icons/data-processing.png")

# -------------------------------------------------------------
# AutoFeat
#
# -------------------------------------------------------------
# Load DataTypeIdentifier class
class DataTypeIdentifier(object):

    def __init__(self, encoder=None, mappings=None):
        # The encoder is used to encode the target variable
        if encoder is not None:
            self.__encoder = encoder()
            self.__encoder_2 = encoder()
        self.__mappings = mappings

    def keep_initial_data_types(self, original_data):
        # Render immutable the data type of every feature that was set before the data was imported
        # through avoiding having integers being transformed into float
        data = original_data.copy(deep=True)
        for column in data.columns:
            try:
                data = data.astype({column: pd.Int64Dtype()})
            except TypeError:
                pass
        return data

    def build_final_set(self, original_correctly_typed_data, target_variable=None):
        # Create is_float and unique_values features to help predict if a feature is numerical or categorical
        correctly_typed_data = original_correctly_typed_data.copy(deep=True)
        correctly_typed_data.dropna(inplace=True)
        new_features_list = []
        for feature_name in correctly_typed_data:
            feature = correctly_typed_data[feature_name]
            # Check feature data type
            is_float = 0
            if feature.dtype == float:
                is_float = 1
            # Compute steps between one modality and the one following it
            feature_encoded = feature
            if feature.dtype == object:
                feature_encoded = self.__encoder_2.fit_transform(feature)
            feature_encoded = sort(list(set(feature_encoded)))
            # Step for two successive modalities
            index = 0
            step = 1
            while step == 1 and index < len(feature_encoded) - 1:
                step = feature_encoded[index + 1] - feature_encoded[index]
                index += 1
            # Affect 1 to "one_step" if every step is constant and equals to 1
            one_step = 0
            if step == 1:
                one_step = 1
            # Count unique values per feature
            unique_values = feature.nunique()
            # Summarize everything in a list for every single feature
            new_features_list.append([is_float, unique_values, one_step])
        new_features_list = array(new_features_list)
        # Scale(mean:0 and std:1) the values in order to keep the big modalities within a certain range
        new_features_list[:, 1] = scale(new_features_list[:, 1])
        # Dataframe depicting the new features
        new_features = pd.DataFrame(new_features_list, columns=["is_float", "unique_values", "one_step"])
        # Encode target variable
        target_variable_encoded = None
        if target_variable is not None:
            target_variable_encoded = pd.DataFrame(self.__encoder.fit_transform(target_variable))
        # Put features and the target variable in one dictionary
        features_target_dict = {'new_features': new_features, 'target_variable_encoded': target_variable_encoded}
        return features_target_dict

    def label_predictions(self, predictions, mappings):
        # Label predictions according to the mappings: 0 for "categorical" and 1 for "numerical"
        labeled_predictions = []
        for prediction in predictions:
            prediction = prediction[0]
            labeled_predictions.append(mappings[prediction])
        return labeled_predictions

    def predict(self, data, mappings, model):
        # Keep the initial data types
        accurately_typed_data = self.keep_initial_data_types(data)
        # Build the final_set for our model, no target variable is given because it is what we are trying to predict
        final_set = self.build_final_set(original_correctly_typed_data=accurately_typed_data)
        # Get the features: "is_float" and "unique_values"
        features = final_set["new_features"]
        # Get our predictions
        predictions = (model.predict(features) > 0.5).astype("int32")
        # Label our predictions: 0 represents "categorical" and 1 represents "numerical"
        labeled_predictions = self.label_predictions(predictions, mappings)
        # Summarize everything in a dataframe
        final_predictions = pd.DataFrame(labeled_predictions, columns=["Data type prediction"], index=data.columns)
        return final_predictions

    def load_variables(self, path):
        # Load variables with pickle.
        loaded_variable = None
        with open(path, "rb") as file:
            loaded_variable = pickle.load(file)
        return loaded_variable

def bokeh_page(doc):
    def auto_mode(variable, cardinality, category):
        if cardinality < 15 and category == "Nominal":
            return "OneHot"
        elif cardinality > 15 and category == "Nominal":
            return "Hash"
        elif cardinality < 15 and category == "Ordinal":
            return "Ordinal"
        elif cardinality > 15 and category == "Ordinal":
            return "Binary"
    def encode_categorical_data(dataset, data):
        data = pd.DataFrame.from_dict(data)
        categorical_dataset = data[data.Type == "categorical"]

        ordinal_enc, one_hot_enc, dummy_enc, binary_enc = [], [], [], []
        base_n_enc, hash_enc, target_enc = [], [], []

        data_encoded = dataset

        for variable in categorical_dataset["Name"]:
            index = int(categorical_dataset.index[categorical_dataset.Name == variable].values)
            cardinality = categorical_dataset["Cardinality"][index]
            category = categorical_dataset["Category"][index]
            coding_method = categorical_dataset["Coding"][index]

            if coding_method == 'Auto':
                auto_coding_method = auto_mode(variable, cardinality, category)
                if auto_coding_method == "OneHot":
                    one_hot_enc.append(variable)
                elif auto_coding_method == "Hash":
                    hash_enc.append(variable)
                elif auto_coding_method == "Ordinal":
                    ordinal_enc.append(variable)
                elif auto_coding_method == "Binary":
                    binary_enc.append(variable)
            elif coding_method == 'Label':
                ordinal_enc.append(variable)
            elif coding_method == 'OneHot':
                one_hot_enc.append(variable)
            elif coding_method == 'Dummy':
                dummy_enc.append(variable)
            elif coding_method == 'Binary':
                binary_enc.append(variable)
            elif coding_method == 'BaseN':
                base_n_enc.append(variable)
            elif coding_method == 'Hash':
                hash_enc.append(variable)
            elif coding_method == 'Target':
                target_enc.append(variable)

        for variable in one_hot_enc:
            # Create object for one-hot encoder
            encoder = ce.OneHotEncoder(cols=variable, handle_unknown='return_nan', return_df=True, use_cat_names=True)
            # Fit and transform Data
            data_encoded = encoder.fit_transform(data_encoded)

        for variable in hash_enc:
            # Create object for hash encoder
            variable_options = list(categorical_dataset["Options"][categorical_dataset.Name == variable])[0]
            if variable_options == '':
                n=8
            else:
                variable_options_dict = ast.literal_eval(variable_options)
                n = variable_options_dict[list(variable_options_dict.keys())[0]]
            encoder = ce.HashingEncoder(cols=variable, n_components=n)
            # Fit and transform data
            data_encoded = encoder.fit_transform(data_encoded)
            for i in range(n):
                data_encoded = data_encoded.rename(columns={"col_" + str(i): variable + "_" + str(i)})

        for variable in ordinal_enc:
            # Create object for label encoder
            encoder = ce.OrdinalEncoder(cols=variable)
            # Fit and transform data
            data_encoded = encoder.fit_transform(data_encoded)

        for variable in binary_enc:
            # Create object for binary encoder
            encoder = ce.BinaryEncoder(cols=variable, return_df=True)
            # Fit and transform data
            data_encoded = encoder.fit_transform(data_encoded)

        for variable in base_n_enc:
            # Create object for base_n encoder
            variable_options = list(categorical_dataset["Options"][categorical_dataset.Name == variable])[0]
            if variable_options == '':
                n=2
            else:
                variable_options_dict = ast.literal_eval(variable_options)
            n = variable_options_dict[list(variable_options_dict.keys())[0]]
            encoder = ce.BaseNEncoder(cols=variable, return_df=True, base=n)
            # Fit and transform data
            data_encoded = encoder.fit_transform(data_encoded)

        for variable in target_enc:
            column_names = list(data_encoded)
            # Create object for target encoder
            variable_options = list(categorical_dataset["Options"][categorical_dataset.Name == variable])[0]
            variable_options_dict = ast.literal_eval(variable_options)
            target_column = variable_options_dict[list(variable_options_dict.keys())[0]]
            encoder = ce.TargetEncoder(cols=variable)
            # Fit and transform data
            data_encoded_prime = encoder.fit_transform(data_encoded[variable], data_encoded[target_column])
            data_encoded.drop(variable, axis=1, inplace=True)
            data_encoded[variable] = data_encoded_prime[variable]
            data_encoded = data_encoded.reindex(columns=column_names)

        for variable in dummy_enc:
            data_encoded = pd.get_dummies(data=data_encoded, columns=[variable], drop_first=True)

        return data_encoded

    def auto_detection(dataset=None):
        dataframe = pd.read_csv(dataset, sep=FILE_DELIMITER)

        # Use LabelEncoder to keep a certain order of modalities
        data_type_identifier = DataTypeIdentifier(LabelEncoder)

        # -------------------------------------------------------------
        # Load the model and the mappings
        data_type_identifier_model = load_model("data_type_identifier/data_type_identifier.h5")
        mappings = data_type_identifier.load_variables("data_type_identifier/mappings.pickle")

        # -------------------------------------------------------------
        # Predict the columns data type
        data_type_predictions = data_type_identifier.predict(dataframe, mappings, data_type_identifier_model)

        # Save categorical Variables into a list
        cat_variables = []
        index = data_type_predictions.index
        number_of_rows = len(index)
        for i in range(number_of_rows):
            if data_type_predictions.iloc[i]['Data type prediction'] == 'categorical':
                cat_variables.append(data_type_predictions.index.values[i])
        dataframe = (data_type_predictions.reset_index()).rename(
            columns={'index': 'Name', 'Data type prediction': 'Type'}, inplace=False)
        return dataframe

    global file_path
    # file_path = 'wine.csv'
    data = auto_detection(file_path)
    dataset = pd.read_csv(file_path, sep=FILE_DELIMITER)
    original_dataframe = pd.read_csv(file_path, sep=FILE_DELIMITER)
    data['Category'] = 'NA'

    for i in range(len(data)):
        try:
            data.loc[data.Name == list(data['Name'])[i], "Max"] = dataset[list(data['Name'])[i]].max()
            data.loc[data.Name == list(data['Name'])[i], "Min"] = dataset[list(data['Name'])[i]].min()
            data.loc[data.Name == list(data['Name'])[i], "Mean"] = round(dataset.mean(axis=0)[list(data['Name'])[i]], 3)
            data.loc[data.Name == list(data['Name'])[i], "Zeros"] = (dataset[list(data['Name'])[i]] == 0).sum()
            data.loc[data.Name == list(data['Name'])[i], "Missing"] = (dataset.isna().sum())[list(data['Name'])[i]]
        except:
            data.loc[data.Name == list(data['Name'])[i], "Max"] = "-"
            data.loc[data.Name == list(data['Name'])[i], "Min"] = "-"
            data.loc[data.Name == list(data['Name'])[i], "Mean"] = "-"
            data.loc[data.Name == list(data['Name'])[i], "Zeros"] = "-"
            data.loc[data.Name == list(data['Name'])[i], "Missing"] = (dataset.isna().sum())[list(data['Name'])[i]]

    data['Cardinality'] = '-'
    dataset = dataset.replace(' ', np.nan)
    variables = list(data.loc[data.Type == "categorical"]["Name"])
    for variable in variables:
        data.loc[data.Name == variable, "Cardinality"] = dataset[variable].unique().size
    data['Coding'] = '-'
    data['Options'] = ""
    data.loc[data.Type == "categorical", "Coding"] = "Auto"
    data.loc[data.Type == "categorical", "Max"] = '-'
    data.loc[data.Type == "categorical", "Min"] = '-'
    data.loc[data.Type == "categorical", "Mean"] = '-'
    data.loc[data.Type == "categorical", "Zeros"] = '-'
    data.loc[data.Type == "numerical", "Cardinality"] = '-'

    # Label_column.txt to save the label column
    f= open("label_column.txt","w+")
    f.write("")
    f.close()

    original_data = data
    tab_id = 1

    # Error.Warning.Info messages
    text_target_column_info = ' <div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem 1.25rem;margin-bottom:1rem;border:1px solid transparent;border-radius:.25rem;color:#0c5460;background-color:#d1ecf1;border-color:#bee5eb"> <strong>Info!</strong> Please click on save to proceed. Then, specify the target column for each of the dataframe columns, one by one.</span></div>'
    text_remaining_target_column_warning = '<div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                                           '1.25rem;margin-bottom:1rem;border:1px solid transparent;border-radius:.25rem; color: ' \
                                           '#9F6000;background-color: #FEEFB3;border-color:#9F6000;"><strong>Warning!</strong> Please ' \
                                           'specify the target column for the remaining dataset features.</span></div> '
    text_target_column_warning = '<div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                                 '1.25rem;margin-bottom:1rem;border:1px solid transparent;border-radius:.25rem; color: ' \
                                 '#9F6000;background-color: #FEEFB3;border-color:#9F6000;"><strong>Warning!</strong> Please ' \
                                 'specify the target column for each of the dataframe columns, one by one.</span></div> '
    text_missing_category_type = '<div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                                 '1.25rem;margin-bottom:1rem;border:1px solid ' \
                                 'transparent;border-radius:.25rem;color:#721c24;background-color:#f8d7da;border-color' \
                                 ':#f5c6cb;"><strong>Error!</strong> Please don\'t forget to specify the category type.</span></div> '
    text_delete_column_info = '<div style="width:100%;position:relative;padding:.75rem ' \
                              '1.25rem;margin-bottom:1rem;border:1px solid ' \
                              'transparent;border-radius:.25rem;color:#155724;background-color:#d4edda;border-color' \
                              ':#c3e6cb"><strong>Success!</strong> The column has been deleted successfully.</div> '
    text_missing_parameters_error = '<div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                                    '1.25rem;margin-bottom:1rem;border:1px solid ' \
                                    'transparent;border-radius:.25rem;color:#721c24;background-color:#f8d7da;border-color:#f5c6cb' \
                                    ';"><strong>Error!</strong> Some of the encoding parameters are missing. Please check if the ' \
                                    'category type is specified for all the categorical features as well.</span></div> '
    text_auto_all_coding_method = '<div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                                  '1.25rem;margin-bottom:1rem;border:1px solid ' \
                                  'transparent;border-radius:.25rem;color:#0c5460;background-color:#d1ecf1;border-color' \
                                  ':#bee5eb"> <strong>Info!</strong> The coding method of all dataset columns is set to ' \
                                  '<strong>Auto</strong> by default.</span></div> '
    text_n_components_default_set = '<div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                                    '1.25rem;margin-bottom:1rem;border:1px solid ' \
                                    'transparent;border-radius:.25rem;color:#0c5460;background-color:#d1ecf1;border' \
                                    '-color:#bee5eb"> <strong>Info!</strong> nComponents is set to <strong>8</strong> ' \
                                    'by default, if no value is specified.</span></div> '
    text_base_n_default_set = '<div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                              '1.25rem;margin-bottom:1rem;border:1px solid ' \
                              'transparent;border-radius:.25rem;color:#0c5460;background-color:#d1ecf1;border' \
                              '-color:#bee5eb"> <strong>Info!</strong> The default Base for Base N is ' \
                              '<strong>2</strong> which is equivalent to Binary Encoding.</span></div> '
    target_encoder_description = '">&#x1F6C8;<div style="font-size: smaller; width:430px;position:absolute; ' \
                                 'background-color: #C8C8C8;  color: #fff;  text-align: center;  border-radius: 6px;  ' \
                                 'padding: 5px 0;  position:absolute;  z-index: 1;  bottom: 150%;  left: 50%;  ' \
                                 'margin-left: -60px;">Target encoder replaces features with a blend of the expected ' \
                                 'value of the target given particular categorical value and the expected value of the ' \
                                 'target over all the training data.</div></div> '
    hash_encoder_description = '">&#x1F6C8;<div style="font-size: smaller; width:400px;position:absolute; ' \
                               'background-color: #C8C8C8;  color: #fff;  text-align: center;  border-radius: 6px;  ' \
                               'padding: 5px 0;  position:absolute;  z-index: 1;  bottom: 150%;  left: 50%;  ' \
                               'margin-left: -60px;"> Hash encoder represents categorical features using the new dimensions after " \
                            "transformation using n_component argument. By default, we use 8 bits.</div></div>'
    base_n_encoder_description = '">&#x1F6C8;<div style="font-size: smaller; width:400px;position:absolute; ' \
                                 'background-color: #C8C8C8;  color: #fff;  text-align: center;  border-radius: 6px;  ' \
                                 'padding: 5px 0;  position:absolute;  z-index: 1;  bottom: 150%;  left: 50%;  ' \
                                 'margin-left: -60px;"> Base-N encoder encodes the categories into arrays of their base-N representation. A base ' \
                                 'of 2 is equivalent to binary encoding.</div></div>'
    file_reading_exception = '<div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                             '1.25rem;margin-bottom:1rem;border:1px solid ' \
                             'transparent;border-radius:.25rem;color:#721c24;background-color:#f8d7da;border-color' \
                             ':#f5c6cb;"><strong>Error!</strong> Permission denied, Cannot open the file ' + file_path + '. Please make sure that the file is not in use by other background applications. </span></div> '

    source = ColumnDataSource(data=data)
    columns = [TableColumn(field="Name", title="Name"), TableColumn(field="Type", title="Type"),
               TableColumn(field="Category", title="Category"), TableColumn(field="Cardinality", title="Cardinality"),
               TableColumn(field="Coding", title="Encoding Method"),
               TableColumn(field="Options", title="Encoding Options")]

    data_table = DataTable(source=source, columns=columns, editable=True, reorderable=False, width_policy='max',
                           height_policy='fixed')

    text_row = TextInput(value="", title="", width=0, disabled=True, visible=False)


    source_code = """
    var grid = document.getElementsByClassName('grid-canvas')[8].children;
    var row, column = '';

    for (var i = 0,max = grid.length; i < max; i++){
        if (grid[i].outerHTML.includes('active')){
            row = i;
            for (var j = 0, jmax = grid[i].children.length; j < jmax; j++)
                if(grid[i].children[j].outerHTML.includes('active')) 
                    { column = j }
        }
    }
    
    text_row.value = String(row);

    """

    callback = CustomJS(args = dict(source = source, text_row = text_row, new_data = list(data['Name']), old_data = list(original_data['Name'])), code = source_code)
    source.selected.js_on_change('indices', callback)
    ##source.selected.on_change('indices', py_callback)

    column_name = Select(
        value='Select a column',
        title='Column Name',
        width=140,
        height=25,
        disabled=False
    )
    columns_names = ['Select a column', 'All']
    dataframe = dict(source.data)
    data_names = list((dataframe['Name']))
    column_name.options = [*columns_names, *data_names]

    text_type = Select(
        options=['Select', 'categorical', 'numerical'],
        value='Select',
        title='Column Type',
        width=140,
        height=25,
        disabled=True
    )
    coding_method = Select(
        options=['Auto', 'Label', 'OneHot', 'Dummy', 'Binary', 'BaseN', 'Hash', 'Target'],
        value='Auto',
        title='Coding Method',
        width=140,
        height=25,
        visible=False
    )

    category_type = RadioButtonGroup(height=30, width=190, margin=(-7, 0, 0, 0), visible=False)

    base = NumericInput(value=None, title="Base", width=150, height=25, visible=False)
    n_components = NumericInput(value=None, title="nComponents", width=150, height=25, visible=False)
    target = Select(
        options=[''],
        value='',
        title='Target Column Name',
        width=150,
        height=25,
        visible=False
    )

    tooltip = Div(text='', visible=False)

    labels = ['Ordinal', 'Nominal']

    def select_row(attr, old, new):
        data = dict(source.data)
        error_message.visible = False
        error_message.text = ''
        text_type.disabled = False
        edit_btn.visible = True
        dataframe = pd.DataFrame.from_dict(source.data)
        new_data = dataframe.reset_index(drop=True)
        original_data['index'] = list(range(len(original_data['Name'])))

        if not pd.DataFrame.to_dict(original_data) == pd.DataFrame.to_dict(new_data):
            restore_btn.visible = True
        target.visible = False
        base.visible = False
        n_components.visible = False
        delete_btn.visible = True
        selected_column = column_name.value

        if selected_column == 'Select a column':
            text_type.visible = False
            category_type.visible = False
            div_category.visible = False
            label_column.visible = False
            coding_method.visible = False
            tooltip.text = ''
            tooltip.visible = False
            edit_btn.visible = False
            delete_btn.visible = False
            target.visible = False
            base.visible = False
            n_components.visible = False

        elif selected_column == 'All':
            text_type.visible = True
            text_type.value = 'numerical'
            category_type.visible = False
            div_category.visible = False
            label_column.visible = False
            coding_method.visible = False
            target.visible = False
            base.visible = False
            n_components.visible = False
            tooltip.text = ''
            tooltip.visible = False
            edit_btn.visible = True
            delete_btn.visible = False
            button.visible = True

        else:
            text_type.visible = True
            if len(data['Name']) == 1:
                delete_btn.disabled = True
            text_row.value = str(pd.DataFrame(data)["Name"].tolist().index(selected_column))
            name = list(data['Name'])[int(text_row.value)]
            old_type = list(data['Type'])[int(text_row.value)]
            old_cat = list(data['Category'])[int(text_row.value)]
            old_coding_method = list(data['Coding'])[int(text_row.value)]
            column_name.value = name
            text_type.value = old_type

            if old_type == 'categorical':
                category_type.labels = labels
                category_type.visible = True
                div_category.visible = True

                with open('label_column.txt') as f:
                    LABEL_COLUMN = f.read()
                f.close()
                if LABEL_COLUMN == selected_column:
                    label_column.active = 0
                else:
                    label_column.active = None
                label_confirm = LABEL_COLUMN
                coding_method.visible = True
                coding_method.value = old_coding_method
                columns_names = ['Select a column', 'All']
                data_names = list((data['Name']))
                column_name.options = [*columns_names, *data_names]

                if old_coding_method == 'Target':
                    target.visible = True
                    target_options = list(data['Name'])
                    target_options.append('')
                    target_options.remove(column_name.value)
                    target.options = target_options
                    old_coding_method = list(data['Options'])[int(text_row.value)]
                    old_coding_method_dict = ast.literal_eval(old_coding_method)
                    option_name = list(old_coding_method_dict.keys())[0]
                    option_value = old_coding_method_dict[option_name]
                    target.value = option_value

                elif old_coding_method == 'Hash':
                    n_components.visible = True
                    old_coding_method = list(data['Options'])[int(text_row.value)]
                    old_coding_method_dict = ast.literal_eval(old_coding_method)
                    option_name = list(old_coding_method_dict.keys())[0]
                    option_value = old_coding_method_dict[option_name]
                    n_components.value = option_value

                elif old_coding_method == 'BaseN':
                    base.visible = True
                    old_coding_method = list(data['Options'])[int(text_row.value)]
                    old_coding_method_dict = ast.literal_eval(old_coding_method)
                    option_name = list(old_coding_method_dict.keys())[0]
                    option_value = old_coding_method_dict[option_name]
                    base.value = option_value
                if old_cat == 'NA':
                    category_type.active = None
                else:
                    category_type.active = labels.index(old_cat)

            elif old_type == 'numerical':
                text_type.value = 'numerical'
                category_type.visible = False
                category_type.active = None
                div_category.visible = False
                coding_method.visible = False
                label_column.active = None
                label_column.visible = False
                n_components.visible = False
                base.visible = False
                target.visible = False
                tooltip.text = ''
                tooltip.visible = False

            else:
                text_type.value = 'Select'
                category_type.visible = False
                div_category.visible = False
                coding_method.visible = False
                label_column.visible = False
                n_components.visible = False
                base.visible = False
                target.visible = False
                tooltip.text = ''
                tooltip.visible = False

    def update_layout(attr, old, new):
        dataframe = pd.DataFrame.from_dict(source.data)
        data = dataframe.reset_index(drop=True)
        column_type = text_type.value
        if column_name.value == 'All' and column_type == 'categorical':
            category_type.visible = True
            div_category.visible = True
            category_type.active = None
            category_type.labels = labels
            coding_method.visible = True
            label_column.visible = False
            coding_method.value = 'Auto'
        elif column_name.value == 'All' and column_type == 'numerical':
            category_type.visible = False
            div_category.visible = False
            coding_method.visible = False
            label_column.visible = False
            n_components.visible = False
            base.visible = False
            target.visible = False
            tooltip.text = ''
            tooltip.visible = False

        elif column_name.value != 'All' and column_name.value != 'Select a column':
            old_type = data['Type'][int(text_row.value)]
            if column_type == 'numerical':
                category_type.visible = False
                div_category.visible = False
                coding_method.visible = False
                label_column.visible = False
                n_components.visible = False
                base.visible = False
                target.visible = False
                tooltip.text = ''
                tooltip.visible = False

            elif column_type == 'categorical':
                if old_type != column_type:
                    category_type.visible = True
                    div_category.visible = True
                    category_type.active = None
                    category_type.labels = labels
                    coding_method.visible = True
                    label_column.visible = True
                    label_column.active = None
                    coding_method.value = 'Auto'
                    variable = data['Name'][int(text_row.value)]
                else:
                    category_type.visible = True
                    div_category.visible = True
                    category = data['Category'][int(text_row.value)]
                    if category != 'NA':
                        category_type.active = labels.index(category)
                    else:
                        category_type.active = None
                    coding_method.visible = True
                    label_column.visible = True
                    coding_method.value = data['Coding'][int(text_row.value)]

        source.data = dict(data)

    def specify_more_encoding_values(attr, old, new):
        coding = coding_method.value
        if coding == 'Target':
            dataframe = pd.DataFrame.from_dict(source.data)
            data = dataframe.reset_index(drop=True)
            columns_names = ['Select a column', 'All']
            data_names = list((data['Name']))
            column_name.options = [*columns_names, *data_names]
            if column_name.value == 'All':
                error_message.text = (error_message.text).replace(text_auto_all_coding_method, '')
                error_message.text += text_target_column_info
                error_message.visible = True
                button.disabled = True
                target.visible = False
                n_components.visible = False
                base.visible = False
                tooltip.text = '<div style="position:relative;overflow:hidden;border-bottom: 1px dotted ' \
                               'black;display: inline-block; " onmouseover="this.style.overflow= '
                tooltip.text += "''"
                tooltip.text += ';" onmouseout="this.style.overflow='
                tooltip.text += "'hidden';"
                tooltip.text += target_encoder_description
                tooltip.visible = True
            elif column_name.value != 'All' and column_name != 'Select a column':
                target_options = list(data['Name'])
                target_options.append('')
                target.value = ''
                target_options.remove(column_name.value)
                target.options = target_options
                base.visible = False
                n_components.visible = False
                target.visible = True
                tooltip.text = '<div style="position:relative;overflow:hidden;border-bottom: 1px dotted ' \
                               'black;display: inline-block; " onmouseover="this.style.overflow= '
                tooltip.text += "''"
                tooltip.text += ';" onmouseout="this.style.overflow='
                tooltip.text += "'hidden';"
                tooltip.text += target_encoder_description
                tooltip.visible = True
        elif coding == 'Hash':
            target.visible = False
            base.visible = False
            n_components.visible = True
            tooltip.text = '<div style="position:relative;overflow:hidden;border-bottom: 1px dotted black;display: ' \
                           'inline-block; " onmouseover="this.style.overflow= '
            tooltip.text += "''"
            tooltip.text += ";\" onmouseout=\"this.style.overflow="
            tooltip.text += '\'hidden\';'
            tooltip.text += hash_encoder_description
            tooltip.visible = True
            target_options = ['']
        elif coding == 'BaseN':
            target.visible = False
            n_components.visible = False
            base.visible = True
            tooltip.text = '<div style="position:relative;overflow:hidden;border-bottom: 1px dotted black;display: ' \
                           'inline-block; " onmouseover="this.style.overflow= '
            tooltip.text += "''"
            tooltip.text += ';" onmouseout="this.style.overflow='
            tooltip.text += '\'hidden\';'
            tooltip.text += base_n_encoder_description
            tooltip.visible = True
            target_options = ['']
        else:
            base.visible = False
            target.visible = False
            tooltip.text = ''
            tooltip.visible = False
            n_components.visible = False
            target_options = ['']

    def update_dataframe(event):
        dataframe = pd.DataFrame.from_dict(source.data)
        data = dataframe.reset_index(drop=True)

        column_type = text_type.value
        selected_column = column_name.value

        if selected_column == 'All':

            if column_type == 'numerical':
                data['Type'] = 'numerical'
                data['Category'] = 'NA'
                data['Cardinality'] = '-'
                data['Coding'] = '-'
                data['Options'] = ''

            elif column_type == 'categorical':
                if category_type.active is None:
                    text = text_missing_category_type
                    error_message.text += text
                    error_message.visible = True
                    button.disabled = True

                else:
                    error_message.text = ''
                    data['Type'] = 'categorical'
                    data['Category'] = labels[category_type.active]
                    coding = coding_method.value
                    data['Coding'] = coding

                    if coding == 'BaseN':
                        if base.value is None:
                            value = 2
                            error_message.text += text_base_n_default_set
                        else:
                            value = base.value
                        data['Options'] = str({"base": value})

                    elif coding == 'Target':
                        data['Options'] = str({"Target": target.value})

                    elif coding == 'Hash':
                        if n_components.value is None:
                            value = 8
                            error_message.text += text_n_components_default_set
                            error_message.visible = True
                        else:
                            value = n_components.value
                        data['Options'] = str({"nComponents": value})

                    else:
                        data['Options'] = ""
                        button.disabled = False

                    if coding == 'Auto':
                        error_message.text += text_auto_all_coding_method
                        error_message.visible = True

                    for i in range(len(data)):
                        variable = data['Name'][i]
                        data['Cardinality'][i] = dataset[variable].unique().size

        elif selected_column != 'All':
            if selected_column not in list(data['Name']):
                df = pd.read_csv(file_path, sep=FILE_DELIMITER)
                df.columns.values[int(text_row.value)] = list(data['Name'])[int(text_row.value)]
                try:
                    df.to_csv(file_path, sep=FILE_DELIMITER, index=False, header=True)
                except OSError:
                    error_message.text = file_reading_exception
            if column_type == 'categorical':
                data['Min'][int(text_row.value)] = '-'
                data['Max'][int(text_row.value)] = '-'
                data['Mean'][int(text_row.value)] = '-'
                data['Zeros'][int(text_row.value)] = '-'
                data['Type'][int(text_row.value)] = 'categorical'
                error_message.text = ''
                cat_type = category_type.active
                coding = coding_method.value
                if selected_column in list(data['Name']) and label_column.active != None:
                    f= open("label_column.txt","w+")
                    f.write(selected_column)
                    f.close()
                elif selected_column not in list(data['Name']) and label_column.active != None:
                    label_confirm.value = list(data['Name'])[int(text_row.value)]
                    with open('label_column.txt') as f:
                        LABEL_COLUMN = f.read()
                    f= open("label_column.txt","w+")
                    if LABEL_COLUMN not in list(data['Name']):
                        f.write('')
                    if label_column.active != None :
                        f.write(list(data['Name'])[int(text_row.value)])
                    f.close()
                if cat_type is None:
                    text = '<div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                           '1.25rem;margin-bottom:1rem;border:1px solid transparent;border-radius:.25rem; color: ' \
                           '#9F6000;background-color: #FEEFB3;border-color:#9F6000;"><strong>Warning!</strong> The category type of ' + \
                           data["Name"][int(text_row.value)] + ' is missing.'+str(category_type.active)+'</span></div>'
                    error_message.text += text
                    error_message.visible = True
                    button.disabled = True

                    if target.value == '' and coding == 'Target':
                        text = '<div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                               '1.25rem;margin-bottom:1rem;border:1px solid transparent;border-radius:.25rem; color: ' \
                               '#9F6000;background-color: #FEEFB3;border-color:#9F6000;"><strong>Warning!</strong> The Target column for encoding ' + \
                               data["Name"][int(text_row.value)] + ' is missing.</span></div>'
                        error_message.text += text
                        error_message.visible = True
                        button.disabled = True

                elif target.value == '' and coding == 'Target':
                    text =  '<div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                            '1.25rem;margin-bottom:1rem;border:1px solid transparent;border-radius:.25rem; color: ' \
                            '#9F6000;background-color: #FEEFB3;border-color:#9F6000;"><strong>Warning!</strong> The Target column for encoding ' + \
                            data["Name"][int(text_row.value)] + ' is missing.</span></div>'
                    error_message.text += text
                    error_message.visible = True
                    button.disabled = True

                else:
                    i = column_name.options.index(selected_column)
                    column_name.options = column_name.options[:i]+[list(data['Name'])[int(text_row.value)]]+column_name.options[i+1:]
                    column_name.value = list(data['Name'])[int(text_row.value)]
                    if label_confirm.value == column_name.value:
                        label_column.active = 0
                    button.disabled = False
                    error_message.text = '<div style="width:100%;position:relative;padding:.75rem ' \
                                         '1.25rem;margin-bottom:1rem;border:1px solid ' \
                                         'transparent;border-radius:.25rem;color:#155724;background-color:#d4edda' \
                                         ';border-color:#c3e6cb"><strong>Success!</strong> The column info has been ' \
                                         'saved successfully.</div> '
                    error_message.visible = True
                    data['Coding'][int(text_row.value)] = coding
                    data['Category'][int(text_row.value)] = labels[cat_type]
                    variable = data['Name'][int(text_row.value)]
                    dataset = pd.read_csv(file_path, sep=FILE_DELIMITER)
                    data['Cardinality'][int(text_row.value)] = dataset[variable].unique().size

                    if coding == 'BaseN':
                        if base.value is None:
                            value = 2
                            text = text_base_n_default_set
                            error_message.text += text
                        else:
                            value = base.value
                        data['Options'][int(text_row.value)] = str({"base": value})

                    elif coding == 'Target':
                        data['Options'][int(text_row.value)] = str({"Target": target.value})

                    elif coding == 'Hash':
                        if n_components.value is None:
                            value = 8
                            error_message.text += text_n_components_default_set
                        else:
                            value = n_components.value
                        data['Options'][int(text_row.value)] = str({"nComponents": value})
                    else:
                        data['Options'][int(text_row.value)] = ""
                    if column_name.value != data['Name'][int(text_row.value)]:
                        data['Name'][int(text_row.value)] = column_name.value
                    if coding == 'Auto':
                        text = '<div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                               '1.25rem;margin-bottom:1rem;border:1px solid ' \
                               'transparent;border-radius:.25rem;color:#0c5460;background-color:#d1ecf1;border-color' \
                               ':#bee5eb"> <strong>Info!</strong> The coding method of ' + \
                               data["Name"][
                                   int(text_row.value)] + ' is set to <strong>Auto</strong> by default.</span></div>'
                        error_message.text += text

            elif column_type == 'numerical':
                dataset = pd.read_csv(file_path, sep=FILE_DELIMITER)
                data['Type'][int(text_row.value)] = 'numerical'
                data['Category'][int(text_row.value)] = 'NA'
                data['Coding'][int(text_row.value)] = '-'
                data['Cardinality'][int(text_row.value)] = '-'
                data['Options'][int(text_row.value)] = ''
                data['Max'][int(text_row.value)] = dataset[list(data['Name'])[int(text_row.value)]].max()
                data['Min'][int(text_row.value)] = dataset[list(data['Name'])[int(text_row.value)]].min()
                data['Mean'][int(text_row.value)] = round(dataset.mean(axis=0)[list(data['Name'])[int(text_row.value)]], 3)
                data['Zeros'][int(text_row.value)] = (dataset[list(data['Name'])[int(text_row.value)]] == 0).sum()
                data['Missing'][int(text_row.value)] = (dataset.isna().sum())[list(data['Name'])[int(text_row.value)]]
                i = column_name.options.index(selected_column)
                column_name.options = column_name.options[:i]+[list(data['Name'])[int(text_row.value)]]+column_name.options[i+1:]
                column_name.value = list(data['Name'])[int(text_row.value)]


        variable_options_dict_list = list()

        for s in list(data['Options']):
            if 'Target' in s:
                variable_options_dict_list.append(ast.literal_eval(s))

        if variable_options_dict_list:
            if all(s['Target'] == '' for s in variable_options_dict_list):
                if text_target_column_info not in error_message.text:
                    error_message.text += text_target_column_warning
                    error_message.visible = True
                    button.disabled = True

            elif any(s['Target'] == '' for s in variable_options_dict_list) and any(
                    s['Target'] != '' for s in
                    variable_options_dict_list) and text_remaining_target_column_warning not in error_message.text:
                error_message.text += text_remaining_target_column_warning
                error_message.visible = True
                button.disabled = True
            else:
                button.disabled = False
        else:
            button.disabled = False
        if all(s == '-' for s in data['Coding']):
            button.disabled = True

        dataframe = data
        source.data = dict(dataframe)


        # Show up restore button after the save if new info is introduced!
        dataframe = pd.DataFrame.from_dict(source.data)
        new_data = dataframe.reset_index(drop=True)
        original_data['index'] = list(range(len(original_data['Name'])))
        if not pd.DataFrame.to_dict(original_data) == pd.DataFrame.to_dict(new_data):
            restore_btn.visible = True

    def delete_column(event):
        dataframe = pd.DataFrame.from_dict(source.data)
        data = dataframe.reset_index(drop=True)

        if column_name.value != 'All' and column_name.value != 'Select a column':
            index = int(text_row.value)
            if len(data["index"]) > 0:
                data = data.drop([index])
                data['index'] = list(range(len(data['index'])))
                df = pd.read_csv(file_path, sep=FILE_DELIMITER)
                df.drop(column_name.value, axis=1, inplace=True)
                try:
                    df.to_csv(file_path, sep=FILE_DELIMITER, index=False, header=True)
                    columns_names = ['Select a column', 'All']
                    data_names = list((data['Name']))
                    column_name.options = [*columns_names, *data_names]
                    edit_btn.visible = False
                    delete_btn.visible = True
                    column_name.value = 'Select a column'
                    error_message.visible = True
                    error_message.text += text_delete_column_info
                    source.data = dict(data)

                    if len(data["index"]) == 1:
                        button.disabled = True

                    # show up restore button after the save if new info is introduced!
                    dataframe = pd.DataFrame.from_dict(source.data)
                    new_data = dataframe.reset_index(drop=True)
                    original_data['index'] = list(range(len(original_data['Name'])))
                    if not pd.DataFrame.to_dict(original_data) == pd.DataFrame.to_dict(new_data):
                        restore_btn.visible = True

                    variable_options_dict_list = list()
                    for s in list(data['Options']):
                        if 'Target' in s:
                            variable_options_dict_list.append(ast.literal_eval(s))
                    if variable_options_dict_list:
                        if all(s['Target'] == '' for s in variable_options_dict_list):
                            if text_target_column_info not in error_message.text:
                                error_message.text = (error_message.text).replace(text_auto_all_coding_method, '')
                                error_message.text += text_target_column_warning
                                button.disabled = True

                        elif any(s['Target'] == '' for s in variable_options_dict_list) and any(
                                s['Target'] != '' for s in variable_options_dict_list):
                            error_message.text += text_remaining_target_column_warning
                            button.disabled = True
                        else:
                            button.disabled = False
                    else:
                        button.disabled = False

                    if all(s == '-' for s in data['Coding']):
                        button.disabled = True
                except OSError:
                    error_message.text = file_reading_exception
        source.data = dict(data)

    def restore_dataframe(event):
        df = original_dataframe
        try:
            df.to_csv(file_path, sep=FILE_DELIMITER, index=False, header=True)
            source.data = dict(original_data)
            columns_names = ['Select a column', 'All']
            data_names = list((data['Name']))
            column_name.options = [*columns_names, *data_names]
            edit_btn.visible = False
            delete_btn.visible = False
            restore_btn.visible = False
            label_column.visible = False
            label_confirm.value = ''
            column_name.value = 'Select a column'
            error_message.text = ''
            f= open("label_column.txt","w+")
            f.write('')
            f.close()
        except OSError:
            error_message.text = file_reading_exception

    def select_column(attr, old, new):
        dataframe = pd.DataFrame.from_dict(source.data)
        data = dataframe.reset_index(drop=True)
        if text_row.value != '':
            name = list(data['Name'])[int(text_row.value)]
            if name in column_name.options:
                column_name.value = name

    def send_old_dataframe(attr, old, new):
        original_dataframe.to_csv(file_path_new, sep=FILE_DELIMITER, index=False, header=True)

    text_row.on_change('value', select_column)
    text_type.on_change('value', update_layout)
    column_name.on_change('value', select_row)
    coding_method.on_change('value', specify_more_encoding_values)
    div = Div(text="""""", margin=2)
    div_category = Div(text="Category Type", margin=6, visible=False)
    label_column = RadioGroup(labels=["Label"], active=None, visible = False, width=45)

    error_message = Div(text='', visible=False)

    edit_btn = Button(label="Save", button_type="success", width=45, visible=False)
    edit_btn.on_click(update_dataframe)

    restore_btn = Button(label="Restore", button_type="success", width=60, visible=False)
    restore_btn.on_click(restore_dataframe)

    delete_btn = Button(label="Delete Column", button_type="danger", width=97, visible=False)
    delete_btn.on_click(delete_column)

    quit_btn = Button(label="Cancel and Quit", button_type="danger", width=45, visible=True)
    quit = TextInput(value="", title="", width=0, disabled=True, visible=False)

    quit_btn.js_on_click(CustomJS(args=dict(quit=quit), code=""" if (confirm("Do you really want to leave? The changes you made will not be saved.") == true) {
                                                        quit.value = "OK abandon!";}"""))
    quit.on_change('value', send_old_dataframe)
    quit.js_on_change('value',CustomJS(args=dict(urls=['/shutdown']), code=""" urls.forEach(url => $.get(url));
                                                                            setTimeout(() => {  window.close(); }, 2000);"""))

    tab_id = NumericInput(value=0, title="", disabled=True, visible=False)


    def display_encoded_dataframe(event):
        # Check whether there is a missing attribute in the dataframe, category type ...
        error_message.text = ''
        data = pd.DataFrame.from_dict(source.data)
        dataframe = data.reset_index(drop=True)
        dataset = pd.read_csv(file_path, sep=FILE_DELIMITER)
        out = dataframe.loc[dataframe.Type == "categorical", "Category"].values.tolist()

        if not any(x == 'NA' for x in out):
            pass
        else:
            error_message.text = text_missing_parameters_error
            error_message.visible = True
            button.disabled = True
        variable_options_dict_list = list()

        for s in list(data['Options']):
            if 'Target' in s:
                variable_options_dict_list.append(ast.literal_eval(s))
        target_columns_to_change = set()
        target_columns_to_replace = set()

        for feature in variable_options_dict_list:
            categorical_target_columns = data[(data.Name == feature['Target']) & (data.Type == 'categorical')]
            if feature['Target'] not in list(data['Name']):
                target_columns_to_replace.add(feature['Target'])
            elif not categorical_target_columns.empty:
                target_columns_to_change.add(list(categorical_target_columns['Name'])[0])

        if target_columns_to_change:
            error_message.text += text_missing_parameters_error
            for target_column in target_columns_to_change:
                error_message.text += '<li>' + target_column + '</li>'
            for target_column in target_columns_to_replace:
                error_message.text += '<li>' + target_column + '</li>'
            error_message.text += '</ul></span></div>'

        if target_columns_to_replace and len(data['Name']) > 1:
            for target_column in target_columns_to_replace:
                error_message.text += '<div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                                      '1.25rem;margin-bottom:1rem;border:1px solid ' \
                                      'transparent;border-radius:.25rem;color:#721c24;background-color:#f8d7da;border' \
                                      '-color:#f5c6cb;"><strong>Error!</strong> The column <strong>' + target_column \
                                      + '</strong> has been deleted. Please select another target column ' \
                                        'instead.</span></div> '

        elif target_columns_to_replace and len(data['Name']) <= 1:
            for target_column in target_columns_to_replace:
                error_message.text += ' <div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem 1.25rem;margin-bottom:1rem;border:1px solid transparent;border-radius:.25rem;color:#721c24;background-color:#f8d7da;border-color:#f5c6cb;"><strong>Error!</strong> The column <strong>' + target_column + '</strong> has been deleted. Target encoder cannot be applied in this case. Please select another encoding method instead.</span></div>'

        error_message.visible = True
        button.disabled = True

        with open('label_column.txt') as f:
            LABEL_COLUMN = f.read()
        label_confirm.value = LABEL_COLUMN
        if LABEL_COLUMN == '':
            error_message.text += '<div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                                  '1.25rem;margin-bottom:1rem;border:1px solid ' \
                                  'transparent;border-radius:.25rem;color:#721c24;background-color:#f8d7da;border' \
                                  '-color:#f5c6cb;"><strong>Error!</strong> Please select the label column' \
                                  '</span></div> '
        if error_message.text == '':
            encoded_data = encode_categorical_data(dataset, dataframe)


            def display_n_encoded_dataframe(attr, old, new):
                dataframe_size = encoded_data.shape[0]
                if n_rows.value == 'All':
                    n = encoded_data.shape[0]
                else:
                    n = int(n_rows.value)
                encoded_data_prime = encoded_data.round(3).head(n)
                encoded_source = ColumnDataSource(data=encoded_data_prime)
                columns = [TableColumn(field=Ci, title=Ci) for Ci in encoded_data.columns]
                encoded_data_table = DataTable(source=encoded_source, columns=columns, editable=False,
                                               reorderable=False, width_policy='max', height_policy='max')
                n_rows_label_pre = Div(text="Show", margin=(30, 0, -10, 0))
                n_rows_label_post = Div(text="entries of " + str(dataframe_size) + " rows", margin=(30, -10, 0, 0))
                actions = row(column(div, cancel_btn), column(div, confirm_btn), column(div, export_btn), confirm)
                filter_datatable = row(n_rows_label_pre, n_rows, n_rows_label_post)
                coding_options = data
                for i in range(len(coding_options)):
                    if coding_options["Coding"][i] == "Auto":
                        coding_options["Coding"][i] = auto_mode(list(coding_options["Name"])[i], list(coding_options["Cardinality"])[i], list(coding_options["Category"])[i])
                coding_parameters = {"Column Name":coding_options["Name"], "Encoding Method":coding_options["Coding"], "Encoding Options":coding_options["Options"]}
                coding_parameters_df = pd.DataFrame.from_dict(coding_parameters)
                coding_parameters_columns = [TableColumn(field=Ci, title=Ci) for Ci in coding_parameters_df.columns]
                source_coding_parameters = ColumnDataSource(coding_parameters)
                coding_parameters_table = DataTable(source=source_coding_parameters, columns=coding_parameters_columns, editable=False)

                layout4 = grid([[filter_datatable, None, None, actions], [encoded_data_table], [None], [coding_parameters_table]], sizing_mode='stretch_width')
                tab4 = Panel(child=layout4, title=title, closable=True)
                structure.tabs[tab_index] = tab4

            def send_new_dataframe(attr, old, new):
                encoded_data.to_csv(file_path_new, sep=FILE_DELIMITER, index=False, header=True)
                coding_parameters_df.to_csv(file_path_params, sep=FILE_DELIMITER, index=False, header=True)

            def delete_encoded_dataframe(event):
                structure.tabs.pop(tab_index)

            def export_dataframe_to_csv(event):
                encoded_data.to_csv('download.csv', sep=FILE_DELIMITER, encoding='utf-8', index=False, header=True)

            n = dataset.shape[0]

            def get_indexes(rows):
                limit = math.floor(math.log(rows, 10)) + 1
                indexes = []
                for i in range(limit):
                    if i > 0:
                        indexes.append(10 ** i)
                return indexes

            options = get_indexes(n)
            options = [str(i) for i in options]
            options.append('All')

            n_rows = Select(
                options=options,
                title='',
                width=60,
                margin=(20, 0, -10, 0))
            n_rows.on_change('value', display_n_encoded_dataframe)
            n_rows_label_pre = Div(text="Show", margin=(30, 0, -10, 0))
            n_rows_label_post = Div(text="entries of " + str(n) + " rows", margin=(30, -10, 0, 0))
            encoded_data_prime = encoded_data.round(3).head(int(options[0]))
            encoded_source = ColumnDataSource(data=encoded_data_prime)
            error_message.text = ' <div style="width:100%;position:relative;padding:.75rem 1.25rem;margin-bottom:1rem;border:1px solid transparent;border-radius:.25rem;color:#155724;background-color:#d4edda;border-color:#c3e6cb"><strong>Success!</strong> The dataframe has been updated successfully.</div>'
            error_message.visible = True
            columns = [TableColumn(field=Ci, title=Ci) for Ci in encoded_data.columns]
            encoded_tab_id = tab_id.value + 1
            encoded_data_table = DataTable(source=encoded_source, columns=columns, editable=False, reorderable=False,
                                           width_policy='max', height_policy='max')
            confirm_btn = Button(label="Proceed", button_type="success", width=90)
            confirm = TextInput(value="", title="", width=0, disabled=True, visible=False)
            confirm_btn.js_on_click(CustomJS(args=dict(confirm_input=confirm), code= """ if (confirm("Do you really want to quit AutoFeat and continue the workflow execution? The changes you made will be saved and transferred to the next connected task, if any exists.") == true) {
                                                                                confirm_input.value = "OK proceed!";}"""))
            confirm.on_change('value',send_new_dataframe)
            confirm.js_on_change('value',CustomJS(args=dict(urls=['/shutdown']), code=""" urls.forEach(url => $.get(url));
                                                                            setTimeout(() => {  window.close(); }, 2000);"""))
            cancel_btn = Button(label="Cancel", button_type="danger", width=90)
            cancel_btn.on_click(delete_encoded_dataframe)
            export_btn = Button(label="Download CSV", button_type="primary", width=90)
            export_btn.on_click(export_dataframe_to_csv)
            export_btn.js_on_click(
                CustomJS(args=dict(urls=['/download']), code="urls.forEach(url => window.open(url))"))
            div = Div(text="""""", margin=2)
            actions = row(column(div, cancel_btn), column(div, confirm_btn), column(div, export_btn), confirm)
            filter_datatable = row(n_rows_label_pre, n_rows, n_rows_label_post)
            coding_options = data
            for i in range(len(coding_options)):
                if coding_options["Coding"][i] == "Auto":
                    coding_options["Coding"][i] = auto_mode(list(coding_options["Name"])[i], list(coding_options["Cardinality"])[i], list(coding_options["Category"])[i])

            coding_parameters = {"Column Name":coding_options["Name"], "Encoding Method":coding_options["Coding"], "Encoding Options":coding_options["Options"]}
            coding_parameters_df = pd.DataFrame.from_dict(coding_parameters)
            coding_parameters_columns = [TableColumn(field=Ci, title=Ci) for Ci in coding_parameters_df.columns]
            source_coding_parameters = ColumnDataSource(coding_parameters)
            coding_parameters_table = DataTable(source=source_coding_parameters, columns=coding_parameters_columns, editable=False)

            layout4 = grid([[filter_datatable, None, None, actions], [encoded_data_table], [None], [coding_parameters_table]], sizing_mode='stretch_width')
            title = 'ENCODED_DATA_' + str(encoded_tab_id)
            tab4 = Panel(child=layout4, title=title, closable=True)
            structure.tabs.append(tab4)
            tab_id.value = encoded_tab_id
            tab_index = structure.tabs.index(tab4)

    button = Button(label="Preview Encoded Data", button_type="primary", width=140)
    button.disabled = True

    error_message.text = ''
    data = pd.DataFrame.from_dict(dataframe)
    categorical_dataset = data[(data.Type == "categorical") & (data.Category == 'NA')]
    for categorical_column in categorical_dataset['Name']:
        error_message.text += ' <div style= "width:100%;padding-right:4rem;position:relative;padding:.75rem ' \
                              '1.25rem;margin-bottom:1rem;border:1px solid transparent;border-radius:.25rem; color: ' \
                              '#9F6000;background-color: #FEEFB3;border-color:#9F6000;"><strong>Warning!</strong> Please ' \
                              'specify the category type of the feature <strong>' + categorical_column + '</strong>.</span></div>'

    error_message.visible = True
    button.on_click(display_encoded_dataframe)
    div_label = Div(text="""""", margin=6)

    label_confirm = TextInput(value="", title="", width=0, disabled=True, visible=False)

    layout = column(
        row(text_row, column_name, text_type, column(div_category, category_type), column(div_label, label_column), coding_method, base, n_components,
            target, tooltip, column(div, edit_btn), column(div, restore_btn), column(div, delete_btn),
            column(div, button), column(div, quit_btn), quit, tab_id, label_confirm), data_table)
    layout1 = grid([[div], [layout], [error_message], ], sizing_mode='stretch_width')

    # Define TAB0's Layout
    def display_n_preview_dataframe(attr, old, new):
        dataset = pd.read_csv(file_path, sep=FILE_DELIMITER)
        if n_rows.value == 'All':
            n = dataset.shape[0]
        else:
            n = int(n_rows.value)
        preview_dataset = dataset.head(n)
        source0 = ColumnDataSource(data=preview_dataset)
        columns0 = [TableColumn(field=Ci, title=Ci) for Ci in preview_dataset.columns]
        data_preview_table = DataTable(source=source0, columns=columns0, editable=False, reorderable=False,
                                       width_policy='max', height_policy='max')
        filter_datatable = row(n_rows_label_pre, n_rows, n_rows_label_post)
        layout0 = grid([[filter_datatable, None, None, None, None, None, actions], [data_preview_table], ],
                       sizing_mode='stretch_width')
        tab0 = Panel(child=layout0, title='DATA PREVIEW', closable=False)
        structure.tabs[0] = tab0

    def get_indexes(rows):
        limit = math.floor(math.log(rows, 10)) + 1
        indexes = []
        for i in range(limit):
            if i > 0:
                indexes.append(10 ** i)
        return indexes

    dataset = pd.read_csv(file_path, sep=FILE_DELIMITER)
    n = dataset.shape[0]
    options = get_indexes(n)
    options = [str(i) for i in options]
    options.append('All')
    n_rows = Select(
        options=options,
        value=options[0],
        title='',
        width=60,
        margin=(20, 0, -10, 0))
    n_rows.on_change('value', display_n_preview_dataframe)
    n_rows_label_pre = Div(text="Show", margin=(30, 0, -10, 0))
    n_rows_label_post = Div(text="entries of " + str(n) + " rows", margin=(30, -10, 0, 0))

    def refresh_datatable(event):
        dataset = pd.read_csv(file_path, sep=FILE_DELIMITER)
        preview_dataset = dataset.head(int(n_rows.value))
        source0 = ColumnDataSource(data=preview_dataset)
        columns0 = [TableColumn(field=Ci, title=Ci) for Ci in preview_dataset.columns]
        data_preview_table = DataTable(source=source0, columns=columns0, editable=False, reorderable=False,
                                       width_policy='max', height_policy='max')
        filter_datatable = row(n_rows_label_pre, n_rows, n_rows_label_post)
        layout0 = grid([[filter_datatable, None, None, None, None, None, actions], [data_preview_table], ],
                       sizing_mode='stretch_width')
        tab0 = Panel(child=layout0, title='DATA PREVIEW', closable=False)
        structure.tabs[0] = tab0

    refresh_btn = Button(label="Refresh", button_type="primary", width=25, visible=True)
    refresh_btn.on_click(refresh_datatable)
    actions = column(div, refresh_btn)
    source0 = ColumnDataSource(data=dataset.head(int(options[0])))
    columns0 = [TableColumn(field=Ci, title=Ci) for Ci in dataset.columns]
    filter_datatable = row(n_rows_label_pre, n_rows, n_rows_label_post)
    data_preview_table = DataTable(source=source0, columns=columns0, editable=False, reorderable=False,
                                   width_policy='max', height_policy='max')
    layout0 = grid([[filter_datatable, None, None, None, None, None, actions], [data_preview_table], ],
                   sizing_mode='stretch_width')

    # Define TAB2's Layout
    columns2 = [TableColumn(field="Name", title="Name"), TableColumn(field="Type", title="Type"),
                TableColumn(field="Category", title="Category"), TableColumn(field="Missing", title="Missing"),
                TableColumn(field="Zeros", title="Zeros"), TableColumn(field="Min", title="Min"),
                TableColumn(field="Max", title="Max"), TableColumn(field="Mean", title="Mean"),
                TableColumn(field="Cardinality", title="Cardinality")]

    layout2 = grid([[div], [
        DataTable(source=source, columns=columns2, editable=False, reorderable=False, width_policy='max',
                  height_policy='max')], ])

    # Define TAB3's Layout
    category = RadioButtonGroup(visible=False)
    category.labels = labels
    category.active = None

    # Define Document Structure
    tab0 = Panel(child=layout0, title='DATA PREVIEW', closable=False)
    tab1 = Panel(child=layout1, title='EDIT COLUMN NAMES AND TYPES', closable=False)
    tab2 = Panel(child=layout2, title='COLUMN SUMMARIES', closable=False)
    structure = Tabs(tabs=[tab0, tab2, tab1])
    doc.add_root(structure)

    doc.theme = Theme(json=yaml.load("""
        attrs:
            Grid:
                grid_line_dash: [6, 4]
                grid_line_color: white
    """, Loader=yaml.FullLoader))

def thread_bokeh():
    server = Server({'/': bokeh_page}, port=bokeh_port, io_loop=IOLoop(), allow_websocket_origin=["*:"+str(flask_port), "*:"+str(bokeh_port)])
    server.start()
    server.io_loop.start()


t_bokehApp = Thread(name='Bokeh App',target=thread_bokeh)
t_bokehApp.setDaemon(True)
t_bokehApp.start()


def thread_flask():
    app = Flask(__name__)

    def shutdown_server():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        open('.stop_server', 'a').close()
        t_bokehApp.join()
        func()

    @app.route('/shutdown', methods=['GET'])
    def shutdown():
        shutdown_server()
        return 'Flask server shutting down...'

    @app.route('/download')
    def download():
        dfbuffer = StringIO()
        df = pd.read_csv('download.csv', sep=FILE_DELIMITER)
        df.to_csv(dfbuffer, sep=FILE_DELIMITER, encoding='utf-8', index=False, header=True)
        dfbuffer.seek(0)
        values = dfbuffer.getvalue()
        return Response(values, mimetype='text/csv')

    @app.route('/', methods=['GET'])
    def home():
        script = server_document(bokeh_url)
        return render_template("index.html", script=script, template="Flask")

    app.run(host='0.0.0.0', port=flask_port, debug=False, use_reloader=False)


t_flaskApp = Thread(name='Flask App', target=thread_flask)
t_flaskApp.setDaemon(True)
t_flaskApp.start()

while not os.path.isfile('.stop_server'):
    time.sleep(1)
os.remove('.stop_server')

print('Flask server stopped.')

# Remove External Endpoint Url
schedulerapi.removeExternalEndpointUrl(variables.get("PA_JOB_ID"), "AutoFeat")

# -------------------------------------------------------------
# Load the new version of the dataframe from bokeh
#
while not os.path.isfile(file_path_new):
    time.sleep(1)
if not os.stat(file_path_new).st_size == 0:
    dataframe = pd.read_csv(file_path_new, sep=FILE_DELIMITER)
else:
    dataframe = pd.read_csv(file_path, sep=FILE_DELIMITER)

# label_column -> column name
with open('label_column.txt') as f:
    LABEL_COLUMN = f.read()

# -------------------------------------------------------------
# Load the selected encoding parameters from bokeh
#
while not os.path.isfile(file_path_params):
    time.sleep(1)
encoding_parameters = ""
encoding_params_str = ""

dataframe_id = compress_and_transfer_dataframe(dataframe)
print("dataframe id (out): ", dataframe_id)

if not os.stat(file_path_params).st_size == 0:
    print("encoding parameters:")
    encoding_parameters_df = pd.read_csv(file_path_params, sep=FILE_DELIMITER)
    encoding_parameters = encoding_parameters_df[encoding_parameters_df["Encoding Method"] != "-"]
    for variable in encoding_parameters["Column Name"]:
        index = int(encoding_parameters.index[encoding_parameters["Column Name"] == variable].values)
        encoding_params_str += "the feature "+variable+" is encoded using the method "+encoding_parameters["Encoding Method"][index]
        if str(encoding_parameters["Encoding Options"][index]) != "nan":
            encoding_params_str += " ("+str(encoding_parameters["Encoding Options"][index])+").\n"
        else:
            encoding_params_str += ".\n"
    print(encoding_params_str)

# -------------------------------------------------------------
# Transfer data to the next tasks
#
resultMetadata.put("task.name", __file__)
resultMetadata.put("task.dataframe_id", dataframe_id)
resultMetadata.put("task.label_column", LABEL_COLUMN)
resultMetadata.put("task.feature_names", feature_names)
resultMap.put("encoding parameters", encoding_parameters)

# -------------------------------------------------------------
# Preview results
#
preview_dataframe_in_task_result(dataframe)

# -------------------------------------------------------------
print("END " + __file__)