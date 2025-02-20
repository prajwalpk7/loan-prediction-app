import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))
#helps to import config.py without error
from prediction_model.config import config

#Load dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(filepath)
    _data.columns = [c.strip() for c in _data.columns] #fix columns
    return _data[config.FEATURES]
#Ignoring loan_id as not mentioned in config.py and load all other cols

#Separate X and y
def separate_data(data):
    X = data.drop(config.TARGET, axis = 1)
    y = data[config.TARGET]
    return X, y

#Split X and y
def split_data(X, y, test_size = 0.2, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    return X_train, X_test, y_train, y_test

#Serialization
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    joblib.dump(pipeline_to_save,save_path)
    print(f"Model has been saved under the name {config.MODEL_NAME}")

#Deserialization
def load_pipeline(pipeline_to_load):
    save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print("Model has been loaded")
    return model_loaded