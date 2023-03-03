import os
import traceback
import logging
import json
import numpy
import joblib

# from azure.storage.blob import BlobClient
import pandas as pd
from io import StringIO, BytesIO


from azureml.core.model import Model
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tensorflow as tf
import mlflow

def get_test_df():
    df = pd.read_csv("https://irisdataversioner.blob.core.windows.net/iris-data/Iris.csv?sp=r&st=2023-03-03T08:28:44Z&se=2023-09-01T16:28:44Z&sv=2021-06-08&sr=b&sig=61N71wu0%2BIDHQf%2B5hQxGngA4RdYmRLhIJ%2BUvJC3dRHA%3D")
    return df

def get_best_model(model_arr):
    """
    This function is called to get the best model from the model array
    """
    df = get_test_df() 

    X = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
    y = df[["Species"]]

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=1984,stratify=y)

    best_model = None
    best_score = 0
    for model in model_arr:
        score = precision_score(y_test, model.predict(X_test), average='weighted')
        if score > best_score:
            best_score = score
            best_model = model
    return best_model


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = f"models:/mlflow_cnn/latest"
    model = mlflow.pyfunc.load_model(model_path)

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("model iris: request received")
    try:
        data = json.loads(raw_data)
        sepal_l_cm = data['SepalLengthCm']
        sepal_w_cm = data['SepalWidthCm']
        petal_l_cm = data['PetalLengthCm']
        petal_w_cm = data['PetalWidthCm']
        test_X = list(zip(sepal_l_cm,sepal_w_cm, petal_l_cm, petal_w_cm) )
        result = model.predict(test_X)
    except Exception as err:
        traceback.print_exc()
    logging.info("Request processed")
    return result.tolist()