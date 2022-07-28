
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
import random 
from Analysis.Tools.Loading import *




# # Splitting the Data to Train, (Validation) and Test Set 
# def TrainTestSplit(image_path,labels={'Positive_Class':'AP - New ROI Data','Negative_Class':'Control - New ROI Data'},
#                     validation=False,get_subjects_from='Folder',image_folder=None,df=None,return_list = True):
#     train = []
#     val = []
#     test = [] 

    
#     if get_subjects_from=='Folder':
#       positive_subjects = sorted(os.listdir(os.path.join(image_path,image_folder,labels['Positive_Class'])))
#       negative_subjects = sorted(os.listdir(os.path.join(image_path,image_folder,labels['Negative_Class'])))
      
#     elif get_subjects_from == 'Raw Data':
#       positive_subjects = sorted(df[df.LABEL == 1]['RECORDING_SESSION_LABEL'].unique().tolist())
#       negative_subjects = sorted(df[df.LABEL == 0]['RECORDING_SESSION_LABEL'].unique().tolist())
     
        
#     r = len(positive_subjects)
#     index= 0
#     first = True
    
#     if validation:
#         for i in range(r):
#             if i == r-1:
#                 positive_val = [positive_subjects[i]]
#                 positive_test = [positive_subjects[0]]
#             else:
#                 positive_val = [positive_subjects[i]]
#                 positive_test = [positive_subjects[i+1]]
          
#             if positive_class == 'DP':
#                 try:
#                   negative_val = [negative_subjects[index],negative_subjects[index+1]]
#                 except IndexError:
#                   negative_val = [negative_subjects[-1]]      
            
#                 try:
#                   negative_test = [negative_subjects[index+2],negative_subjects[index+3]]
#                 except IndexError:
#                   if first:
#                     negative_test = [negative_subjects[-1]]
#                     first = False
#                   else:
#                     negative_test = [negative_subjects[0],negative_subjects[1]]
#                 index += 2
          
#             else:
#                 negative_val = [negative_subjects[index],negative_subjects[index+1]]
#                 negative_test = [negative_subjects[index+2],negative_subjects[index+3]]
#                 index += 2
        
#             positive_train = list(set(positive_subjects).difference(set(positive_test+positive_val)))
#             negative_train = list(set(negative_subjects).difference(set(negative_test+negative_val)))
            
#             train.append({'positive_class':positive_train,'negative_class':negative_train})
#             val.append({'positive_class':positive_val,'negative_class':negative_val})
#             test.append({'positive_class':positive_test,'negative_class':negative_test})
        
#         return train, val, test
    
#     else:
#         index = 0
#         for i in range(r):
#             positive_test = [positive_subjects[i]]
#             negative_test = [negative_subjects[index],negative_subjects[index+1]]
#             index += 2
            
#             positive_train = list(set(positive_subjects).difference(set(positive_test)))
#             negative_train = list(set(negative_subjects).difference(set(negative_test)))
            
#             train.append({'positive_class':positive_train,'negative_class':negative_train})
#             test.append({'positive_class':positive_test,'negative_class':negative_test})
        
#         return train, test






# Splitting the Data to Train, (Validation) and Test Set 
def TrainTestSplit(positive_subjects, negative_subjects, image_path,labels={'Positive_Class':'AP - New ROI Data','Negative_Class':'Control - New ROI Data'},
                    validation=False, return_list = True):
    train = []
    val = []
    test = [] 
     
    r = len(positive_subjects)
    index= 0
    first = True
    
    if validation:
        for i in range(r):
            if i == r-1:
                positive_val = [positive_subjects[i]]
                positive_test = [positive_subjects[0]]
            else:
                positive_val = [positive_subjects[i]]
                positive_test = [positive_subjects[i+1]]
          
            if positive_class == 'DP':
                try:
                  negative_val = [negative_subjects[index],negative_subjects[index+1]]
                except IndexError:
                  negative_val = [negative_subjects[-1]]      
            
                try:
                  negative_test = [negative_subjects[index+2],negative_subjects[index+3]]
                except IndexError:
                  if first:
                    negative_test = [negative_subjects[-1]]
                    first = False
                  else:
                    negative_test = [negative_subjects[0],negative_subjects[1]]
                index += 2
          
            else:
                negative_val = [negative_subjects[index],negative_subjects[index+1]]
                negative_test = [negative_subjects[index+2],negative_subjects[index+3]]
                index += 2
        
            positive_train = list(set(positive_subjects).difference(set(positive_test+positive_val)))
            negative_train = list(set(negative_subjects).difference(set(negative_test+negative_val)))
            
            if return_list:
                train.append(positive_train+negative_train)
                val.append(positive_val+negative_val)
                test.append(positive_test+negative_test)
            else:
                
                train.append({'positive_class':positive_train,'negative_class':negative_train})
                val.append({'positive_class':positive_val,'negative_class':negative_val})
                test.append({'positive_class':positive_test,'negative_class':negative_test})
        
        return train, val, test
    
    else:
        index = 0
        for i in range(r):
            positive_test = [positive_subjects[i]]
            negative_test = [negative_subjects[index],negative_subjects[index+1]]
            index += 2
            
            positive_train = list(set(positive_subjects).difference(set(positive_test)))
            negative_train = list(set(negative_subjects).difference(set(negative_test)))
            
            if return_list:
                train.append(positive_train+negative_train)
                test.append(positive_test+negative_test)
            else:
                train.append({'positive_class':positive_train,'negative_class':negative_train})
                test.append({'positive_class':positive_test,'negative_class':negative_test})
        
        return train, test






def RandomTrainTestSplit(positive_subjects, negative_subjects, percent_val = None,percent_test = 0.3,return_list = True):
  

  train = []
  test = []
  val = []
  

  for i in range(10):

    all_positive_subjects = positive_subjects.copy()
    all_negative_subjects = negative_subjects.copy()

    num_test_subjects = round(percent_test*len(all_positive_subjects))
    if percent_val != None:
      num_val_subjects = round(percent_val*len(all_positive_subjects))

    positive_test_subjects = random.sample(all_positive_subjects, num_test_subjects)
    for subject in positive_test_subjects:
      all_positive_subjects.remove(subject)

    negative_test_subjects = random.sample(all_negative_subjects, num_test_subjects)
    for subject in negative_test_subjects:
      all_negative_subjects.remove(subject) 



    if percent_val != None:
      
        positive_val_subjects = random.sample(all_positive_subjects, num_val_subjects)
        for subject in positive_val_subjects:
            all_positive_subjects.remove(subject)
        
        negative_val_subjects = random.sample(all_negative_subjects, num_val_subjects)
        for subject in negative_val_subjects:
            all_negative_subjects.remove(subject)    
        
        if return_list:
            val.append(positive_val_subjects+negative_val_subjects)
        else:
            val.append({'positive_class':positive_val_subjects,'negative_class':negative_val_subjects})
            
         
    


    if return_list:
        test.append(positive_test_subjects+negative_test_subjects)
        train.append(all_positive_subjects+all_negative_subjects)

    else:
        test.append({'positive_class':positive_test_subjects,'negative_class':negative_test_subjects})
        train.append({'positive_class':all_positive_subjects,'negative_class':all_negative_subjects})


  
  if percent_val != None:
        return train, val, test  
    
  else:
    return train, test





# Training the Classical model on the DataFrame of Scanpaths
def TrainClassifier(data,train_list,test_list,upsample=False):
  
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


