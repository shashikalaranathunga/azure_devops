import traceback
import logging
import json

from azureml.core.model import Model
import tensorflow as tf
import mlflow


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    # model_path = f"models:/mlflow_cnn/latest"
    model_path = Model.get_model_path('mlflow_cnn')
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
        return 1
    except Exception as err:
        traceback.print_exc()
    logging.info("Request processed")
    return 1