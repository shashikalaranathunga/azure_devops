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
from pandas.api.types import is_string_dtype
import re
import math
import subprocess
import joblib

'''
IRIS Dataset Versioning
'''
__author__ = "Lasal Jayawardena"
__email__ = "LasalJ@99x.io"

class IRISDataDriftTester():
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

    def get_latest_dataframe_versions(self, dataset_name, num_versions=2):
        v_dataset = Dataset.get_all(workspace= self.workspace )[dataset_name]
        latest_version = v_dataset.version

        df_list = []
        for i in range(num_versions):
            df = (Dataset.get_by_name(workspace = self.workspace, name = dataset_name,\
                                    version = latest_version-i).to_pandas_dataframe())
            df_list.append(df)

        return df_list


    def calculate_psi(self, previous, current, buckettype='bins', buckets=10, axis=0):
        """
        Calculate the PSI (population stability index) across all variables
        
        Parameters
        ----------
        previous: numpy matrix of original values
        current: numpy matrix of new values, same size as previous
        buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
        buckets: number of quantiles to use in bucketing variables
        axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
        
        Returns
        -------
        psi_values: ndarray of psi values for each variable
        """

        def psi(expected_array, actual_array, buckets):
            """
            Calculate the PSI for a single variable
            
            Parameters
            ----------
            expected_array: numpy array of original values
            actual_array: numpy array of new values, same size as previous
            buckets: number of percentile ranges to bucket the values into
            
            Returns
            -------
            psi_value: calculated PSI value
            """


            def scale_range (input, min, max):
                input += -(np.min(input))
                input /= np.max(input) / (max - min)
                input += min
                return input


            breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

            if buckettype == 'bins':
                breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
            elif buckettype == 'quantiles':
                breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



            expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
            actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

            def sub_psi(e_perc, a_perc):
                """
                Calculate the current PSI value from comparing the values.
                Update the current value to a very small number if equal to zero
                """

                if a_perc == 0:
                    a_perc = 0.0001
                if e_perc == 0:
                    e_perc = 0.0001

                value = (e_perc - a_perc) * np.log(e_perc / a_perc)
                return(value)

            psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

            return(psi_value)

        if len(previous.shape) == 1:
            psi_values = np.empty(len(previous.shape))
        else:
            psi_values = np.empty(previous.shape[axis])

        for i in range(0, len(psi_values)):
            if len(psi_values) == 1:
                psi_values = psi(previous, current, buckets)
            elif axis == 0:
                psi_values[i] = psi(previous[:,i], current[:,i], buckets)
            elif axis == 1:
                psi_values[i] = psi(previous[i,:], current[i,:], buckets)

        return(psi_values)

    def get_psi_dict(self, previous_df, current_df,target):
        features = previous_df.drop(target,axis=1).columns
        psi_dict = {}
        
        for feature in features:
            if is_string_dtype(previous_df[feature]):
                continue
            else:
                psi = calculate_psi(previous_df[feature], current_df[feature], "quantiles")
                psi_dict[feature] = psi
        
        return psi_dict

    def psi_drifted(self, psi_dict, threshold=0.25):
        drifted = False
        for feature, psi in psi_dict.items():
            print(f"Feature '{feature}' has a PSI value of {psi}")
            if psi > threshold:
                drifted = True
                print("Feature '{feature}' has experinced drift")
        return drifted


    def create_pipeline(self):
        '''
        IRIS Dataset Versioning Pipeline
        '''        

        dataset_arr = self.get_latest_dataframe_versions(self.args.dataset_name)

        current_df = dataset_arr[1]
        previous_df = dataset_arr[0]

        psi_dict = self.get_psi_dict(previous_df, current_df, "Species")

        drifted = self.psi_drifted(psi_dict)

        if drifted:

            self.run.tag("Data Drift Detected In IRIS Dataset")    
            self.run.fail("Data Drift Detected In IRIS Dataset")

        else:
            self.run.tag("No Data Drift Detected In IRIS Dataset") 
        self.run.complete()

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Azure ML Dataset Versioning pipeline')
    
    parser.add_argument('--dataset_name', type=str, help='Dataset name to store in workspace')

    args = parser.parse_args()
    iris_ds_drift_tester = IRISDataDriftTester(args)
    iris_ds_drift_tester.create_pipeline()




