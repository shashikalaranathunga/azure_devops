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
import numpy as np
import re
import math
import subprocess
import joblib

'''
IRIS Dataset Versioning
'''
__author__ = "Lasal Jayawardena"
__email__ = "LasalJ@99x.io"

class IRISDataVersioning():
    def __init__(self, args):
        '''
        Initialize Steps
        ----------------
            1. Initalize Azure ML Run Object
        '''
        self.args = args
        self.run = Run.get_context()
        self.workspace = self.run.experiment.workspace

    def dataset_to_update(self):
        '''
        Get the latest data from blob source to update the dataset
        Returns :
            df : Pandas Dataframe with latest data
        '''
        df = pd.read_csv(self.args.blob_sas_url)

        return df


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




