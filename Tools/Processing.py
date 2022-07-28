
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
import statistics
from scipy import stats
import os
import statistics
#from keras.optimizers import Adam
import warnings 
warnings.filterwarnings("ignore")
import statistics
from keras import callbacks



# Processing the list of trials to produce a DataFrame of scanpaths with each scanpath represented as a single row 
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




def XySplit(subjects,path,labels,image_type):
    
    X = []
    y = []
    names = []
    trials = []

    pos_subjects = subjects['positive_class']
    neg_subjects = subjects['negative_class']
    
    for subject in pos_subjects:
        p = f"{path}/{image_folder}/{labels['Positive_Class']}/{subject}"
        subject_images = os.listdir(p)
        random.shuffle(subject_images)
        
        for image in subject_images:
            arr = np.asarray(Image.open(p+'/'+image).convert(f'{image_type}'))
            X.append(arr)
            y.append(1) 
            names.append(subject)
            trial_name = trials.append(image)
            
    for subject in neg_subjects:
        p = f"{path}/{image_folder}/{labels['Negative_Class']}/{subject}"
        subject_images = os.listdir(p)
        random.shuffle(subject_images)
        
        for image in subject_images:
            arr = np.asarray(Image.open(p+'/'+image).convert(f'{image_type}'))
            X.append(arr)
            y.append(0) 
            names.append(subject)
            trial_name = trials.append(image)    
    
    return X, y, names, trials
        
        

# Converting Images Of Scanpaths to Arrays 
class Image2Array:
    
  def __init__(self,subjects,path,image_type,labels):
    self.image_type = image_type
    self.path = path
    X, y, names, trials = XySplit(self.subjects,self.path,self.labels,self.image_type)
    self.X = np.array(X)
    self.y = np.array(y)
    self.names = names
    self.trials = trials
    self.labels = labels
    
  def upsample(self):
    self.X = np.concatenate([self.X,self.X[np.where(self.y == 1)[0]]],axis=0)
    self.y = np.concatenate([self.y,self.y[np.where(self.y == 1)[0]]])

  def shuffle(self):
    shuffler = np.random.permutation(len(self.X))
    self.X = self.X[shuffler]
    self.y = self.y[shuffler]
