
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import random 
import statistics
from scipy import stats
import os
import statistics
from sklearn.metrics import roc_auc_score as aucScore
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from PIL import Image
from sklearn.metrics import roc_auc_score as aucScore
from sklearn.metrics import classification_report
#from keras.optimizers import Adam
import warnings 
warnings.filterwarnings("ignore")
from pylab import rcParams
import statistics
from keras import callbacks
from sklearn.metrics import roc_auc_score as aucScore




# Processing the List of trials to produce a DataFrame of scanpaths with each scanpath represented as a single row 
def Preprocess(allTrials,number_of_fixations):

  roi_dic = {'LeftEyebrow':1, 'RightEyebrow':2, 'LeftEye':3, 'RightEye':4, 'Forehead':5, 'Nose':6, 'Mouth':7, 'Chin':8, 'LeftCheek':9, 'RightCheek':10}
  
  cols =['RECORDING_SESSION_LABEL','CURRENT_FIX_INDEX',f'DURATION_FIRST_{number_of_fixations}', 'DURATION_TOTAL','DURATION_MEAN','DURATION_VARIANCE',
         'PHASE','LABEL','DURATION_FOREHEAD','DURATION_EYEBROWS','DURATION_CHEEKS','DURATION_CHIN']
  
  roi_cols = ['DURATION_EYES','DURATION_NOSE_MOUTH','DURATION_CENTRAL','DURATION_PERIPHERAL']

  data = pd.DataFrame()

  for scp in allTrials:
    scp['CURRENT_FIX_INDEX'] = scp['CURRENT_FIX_INDEX'].map(roi_dic)
    
    features = pd.DataFrame(columns = cols+roi_cols)

    for c in cols:
      features.loc[0,c] = scp.loc[0,c]

    for c in roi_cols: 
      features.loc[0,c] = scp[c].sum()

    for i in range(number_of_fixations):
      features.loc[0,f'Fixation_{i+1}'] = scp.loc[i,'CURRENT_FIX_INDEX']

    data = pd.concat([data,features])
  
  data = data.reset_index(drop=True)
  data = data.drop(columns='CURRENT_FIX_INDEX')

  return data






