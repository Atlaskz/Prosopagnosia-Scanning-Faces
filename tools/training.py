
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score as aucScore
from sklearn.linear_model import LogisticRegression
from tools.loading import *
import random
import os
import json 


def create_splits(positive_subjects, negative_subjects, 
                  p_val_subjects = 0, n_val_subjects = 0, return_list = True):
  
  
  folds = len(positive_subjects)
  train = []
  test = []
  val = []
  vals_used = []
  other = []

  p_indices = np.arange(len(positive_subjects))
  p_splits = list(np.array_split(p_indices, folds))

  n_indices = np.arange(len(negative_subjects))
  n_splits = list(np.array_split(n_indices, folds))
  

  
  
  for f in range(folds):
    
    n_test_indices = list(n_splits[f])
    n_test = [negative_subjects[i] for i in n_test_indices]
    n_other = [elem for i, elem in enumerate(negative_subjects) if i not in n_test_indices]
    
    p_test_indices = list(p_splits[f])
    p_test = [positive_subjects[i] for i in p_test_indices]
    p_other = [elem for i, elem in enumerate(positive_subjects) if i not in p_test_indices]


    p_train, n_train, p_val, n_val = [], [], [], []
    
    if p_val_subjects > 0:
    
      n_train, n_val, vals_used = train_val_split(n_other, n_val_subjects, vals_used)
      p_train, p_val, vals_used = train_val_split(p_other, p_val_subjects, vals_used)

    
    if return_list:
        train.append(p_train+n_train)
        test.append(p_test+n_test)
        val.append(p_val+n_val)
        other.append(p_other+n_other)
    
    else:
        train.append({'positive_class':p_train,'negative_class':n_train})
        test.append({'positive_class':p_test,'negative_class':n_test})
        val.append({'positive_class':p_val,'negative_class':n_val})
        other.append({'positive_class':p_other,'negative_class':n_other})
    
  
  if p_val_subjects > 0:
        return train, val, test
        
  return other, test






def train_val_split(other, num_val_subjects, vals_used):
   
      val = random.sample(other, num_val_subjects)
      while val in vals_used:
          val = random.sample(other,num_val_subjects)
      vals_used.append(val) 
      train = np.setdiff1d(np.array(other),np.array(val)).tolist()

      return train, val, vals_used





# Splitting the Data to Train, (Validation) and Test Set 
def preset_splits(positive_class, positive_subjects, negative_subjects, validation_set = True, return_list = True):
    train = []
    val = []
    test = [] 
     
    r = len(positive_subjects)
    index= 0
    
    if validation_set:
        
        for i in range(r):
            if i == r-1:
                positive_val = [positive_subjects[i]]
                positive_test = [positive_subjects[0]]
            else:
                positive_val = [positive_subjects[i]]
                positive_test = [positive_subjects[i+1]]
            
            if positive_class == 'dp':
                if index >= 12:
                    negative_val = [negative_subjects[index]]
                    negative_test = [negative_subjects[index+1]]
                
                else:
                    negative_val = [negative_subjects[index],negative_subjects[index+1]] 
                    negative_test = [negative_subjects[index+2],negative_subjects[index+3]]
                
                if index >= 12:
                    negative_val = [negative_subjects[index]]
                else:
                    negative_val = [negative_subjects[index],negative_subjects[index+1]]
                
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
            if index >= 12:
                negative_test = [negative_subjects[index]]
                index += 1
            else:
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




# Training the Classical model on the DataFrame of Scanpaths
def train_classifier(data,train_list,test_list,upsample=False):
  
  auc = []
  for i in range(len(test_list)):# corss validation
    
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

      clf = LogisticRegression()
      clf.fit(train_X, train_y)
      preds = clf.predict(test_X)
      auc.append(aucScore(test_y,preds))
# output is the average auc
  return sum(auc)/len(auc)





def save_splits(path):

    positive_subjects = os.listdir(os.path.join(ROOT_PATH,'images',image_folder,'positive_class'))
    negative_subjects = os.listdir(os.path.join(ROOT_PATH,'images',image_folder,'negative_class'))
    negative_subjects = np.setdiff1d(np.array(negative_subjects),np.array(['adcont','cvpcont','fscont', 'wlcont'])).tolist()

    train, val, test = create_splits(positive_subjects, 
                                        negative_subjects, 
                                        p_val_subjects=1,
                                        n_val_subjects=2,
                                        return_list = False)
    
    if not os.path.exists(os.path.join(path)):
        os.mkdir(path)

    with open(os.path.join(path,'train'), 'w') as f:
        json.dump(train, f)

    with open(os.path.join(path,'val'), 'w') as f:
        json.dump(val, f)

    with open(os.path.join(path,'test'), 'w') as f:
        json.dump(test, f)






def load_splits(path):

    with open(os.path.join(path,'train'), 'r') as f:
        train = json.load(f)

    with open(os.path.join(path,'val'), 'r') as f:
        val = json.load(f)

    with open(os.path.join(path,'test'), 'r') as f:
        test = json.load(f)

    return train, val, test