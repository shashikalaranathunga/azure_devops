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

def get_test_df():
    # CONNECTIONSTRING = "DefaultEndpointsProtocol=https;AccountName=irisworkstorage7c89faa70;AccountKey=bRffNtnDwuDZC0ipKZ41QX/tcTzzKCEx6NmT2qpk7ytn3Hlsm/CTVVgPncuJIf2OjugwYf0TOod0+AStl2JyUg==;EndpointSuffix=core.windows.net"
    # CONTAINERNAME= "azureml-blobstore-741f0fc5-f78a-43dc-a80d-11429fa307fa"
    # BLOBNAME= "irisdata/Iris.csv"

    # blob = BlobClient.from_connection_string(conn_str=CONNECTIONSTRING, container_name=CONTAINERNAME, blob_name=BLOBNAME)

    # blob_data = blob.download_blob().readall()
    # df = pd.read_csv(BytesIO(blob_data))
    df = pd.read_csv("https://irisdataversioner.blob.core.windows.net/iris-original/iris_original.csv?sp=r&st=2022-11-04T10:19:36Z&se=2023-01-01T18:19:36Z&sv=2021-06-08&sr=b&sig=K6NFJz6pA51tH7eYQnhG5HPHnwS6u7V55LQ1bPMjYlo%3D")
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
    dt_model_path = Model.get_model_path('IRIS_DT')
    svm_model_path = Model.get_model_path('IRIS_SVM')    
    # deserialize the model file back into a sklearn model
    dt_model = joblib.load(dt_model_path+"/"+"dt_iris_model.pkl")
    svm_model = joblib.load(svm_model_path+"/"+"svm_iris_model.pkl")
    
    model = get_best_model([dt_model, svm_model])
    print("Best model type", type(model).__name__)
    logging.info("Init complete")
    print('IRIS model loaded...')

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