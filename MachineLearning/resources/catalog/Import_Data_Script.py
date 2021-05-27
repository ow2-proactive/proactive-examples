
# -*- coding: utf-8 -*-
"""Proactive Import Data for Machine Learning

This module contains the Python script for the Import Data task.
"""
import ssl
import pickle
import pandas as pd
import urllib.request
from numpy import sort, array
from sklearn.preprocessing import scale
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

global variables, resultMetadata

__file__ = variables.get("PA_TASK_NAME")
print("BEGIN " + __file__)

# -------------------------------------------------------------
# Import an external python script containing a collection of
# common utility Python functions and classes
PA_CATALOG_REST_URL = variables.get("PA_CATALOG_REST_URL")
PA_PYTHON_UTILS_URL = PA_CATALOG_REST_URL + "/buckets/machine-learning/resources/Utils_Script/raw"
if PA_PYTHON_UTILS_URL.startswith('https'):
    exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL, context=ssl._create_unverified_context()).read(), globals())
else:
    exec(urllib.request.urlopen(PA_PYTHON_UTILS_URL).read(), globals())
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
LABEL_COLUMN = variables.get("LABEL_COLUMN")

assert_not_none_not_empty(IMPORT_FROM, "IMPORT_FROM should be defined!")
assert_not_none_not_empty(FILE_PATH, "FILE_PATH should be defined!")
assert_not_none_not_empty(FILE_DELIMITER, "FILE_DELIMITER should be defined!")

# -------------------------------------------------------------
# Load file
#
dataframe = import_csv_file(FILE_PATH, FILE_DELIMITER, IMPORT_FROM)
feature_names = dataframe.columns

# -------------------------------------------------------------
# Transfer data to the next tasks
#
dataframe_id = compress_and_transfer_dataframe(dataframe)
print("dataframe id (out): ", dataframe_id)

resultMetadata.put("task.name", __file__)
resultMetadata.put("task.dataframe_id", dataframe_id)
resultMetadata.put("task.label_column", LABEL_COLUMN)
resultMetadata.put("task.feature_names", feature_names)

# -------------------------------------------------------------
# Preview results
#
preview_dataframe_in_task_result(dataframe)

# -------------------------------------------------------------
# Check if the Data Type Identification is enabled or not
#
data_type_identification_enabled = str_to_bool(variables.get("DATA_TYPE_IDENTIFICATION"), none_is_false=True)

if data_type_identification_enabled:
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
                while step == 1 and index<len(feature_encoded)-1:
                    step = feature_encoded[index+1] - feature_encoded[index]
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
            new_features_list[:,1] = scale(new_features_list[:,1])
            # Dataframe depicting the new features
            new_features = pd.DataFrame(new_features_list, columns=["is_float", "unique_values", "one_step"])
            # Encode target variable
            target_variable_encoded = None
            if target_variable is not None: 
                target_variable_encoded = pd.DataFrame(self.__encoder.fit_transform(target_variable))
            # Put features and the target variable in one dictionary
            features_target_dict = {'new_features': new_features, 'target_variable_encoded':target_variable_encoded}
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
            print(final_predictions)
            return final_predictions

        def load_variables(self, path):
            # Load variables with pickle.
            loaded_variable = None
            with open(path, "rb") as file:
                loaded_variable = pickle.load(file)
            return loaded_variable

    # Use LabelEncoder to keep a certain order of modalities
    data_type_identifier = DataTypeIdentifier(LabelEncoder)

    # -------------------------------------------------------------
    # Load the model and the mappings
    data_type_identifier_model = load_model("data_type_identifier/data_type_identifier.h5")
    mappings = data_type_identifier.load_variables("data_type_identifier/mappings.pickle")

    # -------------------------------------------------------------
    # Predict the columns data type
    data_type_predictions = data_type_identifier.predict(dataframe, mappings, data_type_identifier_model)

    # -------------------------------------------------------------
    # Transfer data to the next tasks
    #
    data_type_dataframe_id = compress_and_transfer_dataframe(data_type_predictions)
    print("data type dataframe id (out): ", data_type_dataframe_id)
    resultMetadata.put("task.data_type_dataframe_id", data_type_dataframe_id)

# -------------------------------------------------------------
print("END " + __file__)