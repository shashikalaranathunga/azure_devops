import traceback
import logging
import json

from azureml.core.model import Model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow

def get_test_df():

    df = pd.read_csv("https://irisdataversioner.blob.core.windows.net/iris-original/iris_original.csv?sp=r&st=2023-03-07T06:17:08Z&se=2023-07-01T14:17:08Z&sv=2021-06-08&sr=b&sig=5JjD%2Bp%2FSi4Tv%2F%2BoGyHmaIyNyiaxmH%2BBv7K%2FK%2F6QROGI%3D")
    return df

def get_best_model(model_arr):
    """
    This function is called to get the best model from the model array
    """
    df = get_test_df() 

    X = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
    y = df[["Species"]]

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=1984,stratify=y)

    return model_arr[0]

    best_model = None
    best_score = 0
    for model in model_arr:
        if isinstance(model, tf.keras.Model):
            preds = model.predict(np.array(X_test)[..., np.newaxis])
            y_pred = np.argmax(preds, axis=1)
            score = precision_score(y_test, list(y_pred), average='weighted')
        else:
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
    global is_tf
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    # model_path = f"models:/mlflow_cnn/latest"
    model_cnn_path = Model.get_model_path('mlflow_cnn')
    model_cnn = mlflow.pyfunc.load_model(model_cnn_path)
    model_svm_path = Model.get_model_path('mlflow_svm')
    model_svm = mlflow.pyfunc.load_model(model_svm_path)
    model_dt_path = Model.get_model_path('mlflow_dt')
    model_dt = mlflow.pyfunc.load_model(model_dt_path)

    model = get_best_model([model_cnn, model_svm, model_dt])
    is_tf = isinstance(model, tf.keras.Model)

    if is_tf:
        print("Best model type - Tensorflow CNN")
    else:
        print("Best model type", type(model).__name__)

    logging.info("Init complete")

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

        if is_tf:
            return 2
            preds = model.predict(np.array(test_X)[..., np.newaxis]) 
            return list(np.argmax(preds, axis=1))
        else:
            retun 1
            return model.predict(test_X)
    except Exception as err:
        traceback.print_exc()
    logging.info("Request processed")
    return "Error"