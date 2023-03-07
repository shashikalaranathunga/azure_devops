import os
import sys

import azureml as aml
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.core.run import Run
import argparse
import json
import time
#import traceback
import logging

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
import re
import math
import seaborn as sn
import matplotlib.pyplot as plt
import subprocess
import joblib

import mlflow

'''
IRIS Classification
'''
__author__ = "Lasal Jayawardena"
__email__ = "LasalJ@99x.io"

class IRISClassification():
    def __init__(self, args):
        '''
        Initialize Steps
        ----------------
            1. Initalize Azure ML Run Object
            2. Create directories
        '''
        self.args = args
        self.run = Run.get_context()
        self.workspace = self.run.experiment.workspace
        os.makedirs('./model_metas', exist_ok=True)

    def get_latest_dataset_version(self, dataset_name):
        '''
        Get the latest version of the Azure ML dataset
        Args :
            dataset_name : name of the Azure ML Dataset
        Returns :
            data_set : Azure ML Dataset object
        '''
        dataset = Dataset.get_by_name(workspace = self.workspace, name = dataset_name, \
                                        version = "latest")

        return dataset


    def get_files_from_datastore(self, container_name, file_name):
        '''
        Get the input CSV file from workspace's default data store
        Args :
            container_name : name of the container to look for input CSV
            file_name : input CSV file name inside the container
        Returns :
            data_ds : Azure ML Dataset object
        '''
        datastore_paths = [(self.datastore, os.path.join(container_name,file_name))]
        data_ds = Dataset.Tabular.from_delimited_files(path=datastore_paths)
        dataset_name = self.args.dataset_name     
        if dataset_name not in self.workspace.datasets:
            data_ds = data_ds.register(workspace=self.workspace,
                        name=dataset_name,
                        description=self.args.dataset_desc,
                        tags={'format': 'CSV'},
                        create_new_version=True)
        else:
            print('Dataset {} already in workspace '.format(dataset_name))
        return data_ds  

    def train_model(self, X_train, X_test, y_train, y_test, model):
        # Early Stopping to prevent overfitting 
        clf = make_pipeline(StandardScaler(), SVC(kernel='linear', gamma='auto'))
        clf.fit(X_train, y_train)

        return clf  

    def create_pipeline(self):
        '''
        IRIS Data training and Validation
        '''        
        # self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        # print("Received datastore")
        # input_ds = self.get_files_from_datastore(self.args.container_name,self.args.input_csv)
        # final_df = input_ds.to_pandas_dataframe()
        input_ds = self.get_latest_dataset_version(self.args.dataset_name)
        final_df = input_ds.to_pandas_dataframe()
        print("Input DF Info",final_df.info())
        print("Input DF Head",final_df.tail())

        X = final_df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
        y = final_df[["Species"]]

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=1984,stratify=y)
        
        with mlflow.start_run(run_name="mlflow_test"):
            # Automatically capture the model's parameters, metrics, artifacts,
            # and source code with the `autolog()` function
            mlflow.sklearn.autolog(log_models=True)

            model = self.train_model(X_train, X_test, y_train, y_test)
            # mlflow.tensorflow.log_model(model, artifact_path="model")
            run_id = mlflow.active_run().info.run_id
            mlflow.sklearn.save_model(model, "./svm_model")

        y_pred = model.predict(X_test)
        print(f"Model Accuracy:{accuracy_score(y_test, y_pred)*100}%")

        model_name = "mlflow_svm"
        # mlflow.register_model(f"runs:/{run_id}/model", model_name)
        model_local_path = "./svm_model"
        mlflow.register_model(f"file://{model_local_path}", model_name)

        self.validate(y_test, y_pred, X_test)

        self.run.complete()

    def create_confusion_matrix(self, y_true, y_pred, name):
        '''
        Create confusion matrix
        '''
        try:
            confm = confusion_matrix(y_true, y_pred, labels=np.unique(y_pred))
            print("Shape : ", confm.shape)

            df_cm = pd.DataFrame(confm, columns=np.unique(y_true), index=np.unique(y_true))
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'
            df_cm.to_csv(name+".csv", index=False)
            self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")

            plt.figure(figsize = (120,120))
            sn.set(font_scale=1.4)
            c_plot = sn.heatmap(df_cm, fmt="d", linewidths=.2, linecolor='black',cmap="Oranges", annot=True,annot_kws={"size": 16})
            plt.savefig("./outputs/"+name+".png")
            self.run.log_image(name=name, plot=plt)
        except Exception as e:
            #traceback.print_exc()    
            logging.error("Create consufion matrix Exception")

    def create_outputs(self, y_true, y_pred, X_test, name):
        '''
        Create prediction results as a CSV
        '''
        pred_output = {"Actual Species" : y_true['Species'].values, "Predicted Species": y_pred['Species'].values}        
        pred_df = pd.DataFrame(pred_output)
        pred_df = pred_df.reset_index()
        X_test = X_test.reset_index()
        final_df = pd.concat([X_test, pred_df], axis=1)
        final_df.to_csv(name+".csv", index=False)
        self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")

    def validate(self, y_true, y_pred, X_test):
        self.run.log(name="Precision", value=round(precision_score(y_true, y_pred, average='weighted'), 2))
        self.run.log(name="Recall", value=round(recall_score(y_true, y_pred, average='weighted'), 2))
        self.run.log(name="Accuracy", value=round(accuracy_score(y_true, y_pred), 2))

        self.create_confusion_matrix(y_true, y_pred, "confusion_matrix")

        y_pred_df = pd.DataFrame(y_pred, columns = ['Species'])
        self.create_outputs(y_true, y_pred_df,X_test, "predictions")
        self.run.tag("IRISClassifierFinalRun")        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QA Code Indexing pipeline')
    parser.add_argument('--container_name', type=str, help='Path to default datastore container')
    parser.add_argument('--input_csv', type=str, help='Input CSV file')
    parser.add_argument('--dataset_name', type=str, help='Dataset name to store in workspace')
    parser.add_argument('--dataset_desc', type=str, help='Dataset description')
    parser.add_argument('--model_path', type=str, help='Path to store the model')
    parser.add_argument('--artifact_loc', type=str, 
                        help='DevOps artifact location to store the model', default='')
    args = parser.parse_args()
    iris_classifier = IRISClassification(args)
    iris_classifier.create_pipeline()