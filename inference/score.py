import json
from mlflow.deployments import get_deploy_client

from azureml.core.model import Model
import traceback
import logging
#from azureml.monitoring import ModelDataCollector

import tensorflow as tf
import mlflow

def init():
    print("This is init")
    global model
    model_path = Model.get_model_path('mlflow_cnn')
    #model_path='azureml://locations/eastus/workspaces/54646eea-ce59-4cf1-9143-dad3c7f31661/models/mlflow_cnn/versions/5'
    model=mlflow.pyfunc.load_model(model_path)


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