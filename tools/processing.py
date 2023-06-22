
import pandas as pd
import numpy as np
import random 
import os
from PIL import Image



# Processing the list of trials to produce a DataFrame of scanpaths with each scanpath represented as a single row 
def preprocess(all_trials,number_of_fixations):

  roi_dic = {'LeftEyebrow':1, 'RightEyebrow':2, 'LeftEye':3, 'RightEye':4, 'Forehead':5, 'Nose':6, 'Mouth':7, 'Chin':8, 'LeftCheek':9, 'RightCheek':10}
  
  cols =['RECORDING_SESSION_LABEL','CURRENT_FIX_INDEX',f'DURATION_FIRST_{number_of_fixations}', 'DURATION_TOTAL','DURATION_MEAN','DURATION_VARIANCE',
         'PHASE','LABEL','DURATION_FOREHEAD','DURATION_EYEBROWS','DURATION_CHEEKS','DURATION_CHIN']
  
  roi_cols = ['DURATION_EYES','DURATION_NOSE_MOUTH','DURATION_CENTRAL','DURATION_PERIPHERAL']

  data = pd.DataFrame()

  for scp in all_trials:
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




def xy_split(subjects,path,image_folder):
    
    X = []
    y = []
    names = []
    trials = []

    pos_subjects = subjects['positive_class']
    neg_subjects = subjects['negative_class']
    
    for subject in pos_subjects:
        p = os.path.join(path,'images',image_folder,'positive_class',subject)
        subject_images = os.listdir(p)
        random.shuffle(subject_images)
        
        for image in subject_images:
            arr = np.asarray(Image.open(p+'/'+image).convert('RGB'))
            X.append(arr)
            y.append(1) 
            names.append(subject)
            trials.append(image.strip('.png'))
            
    for subject in neg_subjects:
        p = os.path.join(path,'images',image_folder,'negative_class',subject)
        subject_images = os.listdir(p)
        random.shuffle(subject_images)
        
        for image in subject_images:
            arr = np.asarray(Image.open(p+'/'+image).convert('RGB'))
            X.append(arr)
            y.append(0) 
            names.append(subject)
            trials.append(image.strip('.png'))
    
    return X, y, names, trials
        
        

# Converting Images Of Scanpaths to Arrays 
class Image2Array:
    
  def __init__(self,subjects,path,labels,image_folder,num_fixations):
    self.subjects = subjects
    self.path = path
    self.labels = labels
    self.image_folder = image_folder
    self.num_fixations = num_fixations
    X, y, names, trials = xy_split(self.subjects,self.path,self.image_folder)
    self.X = np.array(X)
    self.y = np.array(y)
    self.names = names
    self.trials = trials
    
    
  def upsample(self):
    self.X = np.concatenate([self.X,self.X[np.where(self.y == 1)[0]]],axis=0)
    self.y = np.concatenate([self.y,self.y[np.where(self.y == 1)[0]]])

  def shuffle(self):
    shuffler = np.random.permutation(len(self.X))
    self.X = self.X[shuffler]
    self.y = self.y[shuffler]
