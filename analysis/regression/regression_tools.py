
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import random 
from scipy import stats
import os
import statistics
from tensorflow import keras
from sklearn.metrics import roc_auc_score as aucScore
from sklearn.metrics import classification_report
import warnings 
warnings.filterwarnings("ignore")
from analysis.analysis_tools.training import *
from analysis.analysis_tools.loading import *
from analysis.analysis_tools.processing import *
import json


def TrainClassifier(data: pd.core.frame.DataFrame,train_list: list,test_list:list,upsample: bool =False):
    """
    Training a classifier on the first m fixations of the eye tracking data.
    
    Args:
        
        data: trial data 
        train_list: list containing k lists of subjects to train on where k = number of cross-validation folds
        test_list: list containing k lists of subjects to test on 
        upsample: whether to augment the trials in the positive class
    
    Outpout:
        dictionary of auc scores for each of k folds
    
    """
    auc = []
    for i in range(len(test_list)):
    # corss validation
      test_df = data.merge(pd.DataFrame(test_list[i],columns=['RECORDING_SESSION_LABEL']),on='RECORDING_SESSION_LABEL')
      train_df = data.merge(pd.DataFrame(train_list[i],columns=['RECORDING_SESSION_LABEL']),on='RECORDING_SESSION_LABEL')
    
      test_df = test_df.drop(columns=['RECORDING_SESSION_LABEL'])
      train_df = train_df.drop(columns=['RECORDING_SESSION_LABEL'])
    
      # shuffle
      train_df = train_df.sample(frac=1)
      test_df = test_df.sample(frac=1)
    
      train_X = train_df.drop(columns=['LABEL'])
      train_y = train_df['LABEL'].astype(float)
      test_X = test_df.drop(columns=['LABEL'])
      test_y = test_df['LABEL'].astype(float)
    
      # upsample training data with copies of data for the positive label
      if upsample:
        train_X = pd.concat([train_X,train_X[train_X.index.isin(train_y[train_y == 1].index)]])
        train_y = pd.concat([train_y,train_y[train_y == 1]])
    
      clf = LogisticRegression()
      clf.fit(train_X, train_y)
      preds = clf.predict(test_X)
      auc.append(aucScore(test_y,preds))
    # output is the average auc
    return sum(auc)/len(auc)




def RegressionByFixation(labels:dict,
                         max_fixations:int,
                         data_path:str,
                         results_path:str,
                         phases:list =['Learning','Target','Distractor']):
  
    """
    Loading eyetracking data and transforming it to trials of length m fixations (for 1<m=<20) saving the results as a json file
    
    Args:
    
        labels: dict with keys as positive_class and/or negative_class and values as file names for the positive anor negative classe
        max_fixations: max number of fixations for which classification is performed
        data_path: path where eyetracking data is saved
        results_path: path where the results are saved,
        phases: phases of the experiment to load data for
    
    """
  
    # import data from csv files 

    
    #load data
    data = EyeTrackingData(labels, data_path = data_path,phases=phases).Load() 
    
    positive_subjects = sorted(list(data[data.LABEL==1]['RECORDING_SESSION_LABEL'].unique())) #list of positive subjects
    negative_subjects = sorted(list(data[data.LABEL==0]['RECORDING_SESSION_LABEL'].unique())) #list of negative subjects
    
    positive_class = labels['positive_class']
    # split subject data for cross validated training
    train_subjects, test_subjects = CreateSplits(positive_subjects=positive_subjects,
                                              negative_subjects=negative_subjects,
                                              positive_class = positive_class,
                                                validation=False
                                                )
    
    # train models for 1<fixations<=20
    results = {'num_fixations':[],'score':[]}
    
    for num_fixations in range(2,max_fixations+1):
        print(f'using {num_fixations} fixations (max={max_fixations})')
        
        trial_data = LoadTrials(data,num_fixations) # load trials (first n fixations of each trial)
        trial_data = Preprocess(trial_data,num_fixations) # feature engineering 
        # record the fixation number and auc for the plot
        
        auc = TrainClassifier(trial_data, train_subjects, test_subjects,upsample=True)
        
        results['num_fixations'].append(num_fixations)
        results['score'].append(auc) 
    
    
    
    # save results
    with open(os.path.join(results_path,f'{positive_class}_VS_CONTROL'), "w") as write_file:
        json.dump(results, write_file)