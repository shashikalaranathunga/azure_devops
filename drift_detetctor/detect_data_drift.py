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


import pandas as pd
from pandas.api.types import is_string_dtype
import numpy as np
import re
import math
import subprocess
import joblib

'''
IRIS Dataset Drift Detection
'''
__author__ = "Lasal Jayawardena"
__email__ = "LasalJ@99x.io"

class IRISDataDriftDetection():
    def __init__(self, args):
        '''
        Initialize Steps
        ----------------
            1. Initalize Azure ML Run Object
        '''
        self.args = args
        self.run = Run.get_context()
        self.workspace = self.run.experiment.workspace

    def dataset_get_latest_two_versions(self):
        '''
        Get the latest two versions of the dataset data
        Returns :
            df1 : Pandas Dataframe with latest data
            df2 : Pandas Dataframe with one before latest data
        '''
        v_dataset = Dataset.get_all(workspace=Workspace.from_config())[self.args.dataset_name]
        latest_version = v_dataset.version

        df_array = []
        for i in range(2):
            df_array.append( (Dataset.get_by_name(workspace = Workspace.from_config(), name = dataset_name,\
                                    version = latest_version-i).to_pandas_dataframe())
            )

        return df_array


    def create_pipeline(self):
        '''
        IRIS Dataset Versioning Pipeline
        '''        

        if self.args.version:

            print("Starting Versioning Pipeline")

            dataframe = self.dataset_to_update()
            datastore = self.workspace.get_default_datastore()

            Dataset.Tabular.register_pandas_dataframe(dataframe=dataframe, target=datastore, name=self.args.dataset_name)

            self.run.tag("Updated IRIS Dataset Version")    

        else:
            self.run.tag("No Update To IRIS Dataset Version") 
        self.run.complete()

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Azure ML Dataset Versioning pipeline')
    
    parser.add_argument('--dataset_name', type=str, help='Dataset name to store in workspace')
    parser.add_argument('--dataset_desc', type=str, help='Dataset description')
    parser.add_argument('--version', dest='version', action='store_true')
    parser.add_argument('--no-version', dest='version', action='store_false')

    parser.add_argument('--blob_sas_url', type=str, help='SAS URL to the Data File in Blob Storage')

    parser.set_defaults(version=True)
    args = parser.parse_args()
    iris_ds_versioner = IRISDataVersioning(args)
    iris_ds_versioner.create_pipeline()




