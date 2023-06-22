
import pandas as pd
import os
from sklearn.metrics import roc_auc_score as aucScore
from tools.training import *
from tools.loading import *
from tools.processing import *
import json
import plotly.graph_objects as go


def train_classifier(data: pd.core.frame.DataFrame,train_list: list,test_list:list,upsample: bool =False):
    """
    Training a classifier on the first m fixations of the eye tracking data.
    
    Args:
        
        data: trial data 
        train_list: list containing k lists of subjects to train on where k = number of cross-validation folds
        test_list: list containing k lists of subjects to test on 
        upsample: whether to augment the trials in the positive class
    
    Outpout:
        average auc for the k folds
    
    """
    auc = []
    for i in range(len(test_list)): # cross validation
    
      # dataframe of test subjects info
      test_df = data.merge(pd.DataFrame(test_list[i],columns=['RECORDING_SESSION_LABEL']),
                           on='RECORDING_SESSION_LABEL')
      # dataframe of train subjects info
      train_df = data.merge(pd.DataFrame(train_list[i],columns=['RECORDING_SESSION_LABEL']),
                            on='RECORDING_SESSION_LABEL')
      

      print(test_df.RECORDING_SESSION_LABEL.unique())
      test_df = test_df.drop(columns=['RECORDING_SESSION_LABEL'])
      train_df = train_df.drop(columns=['RECORDING_SESSION_LABEL'])
      
      # shuffle the data
      train_df = train_df.sample(frac=1)
      test_df = test_df.sample(frac=1)
      
      print('positive train trials:',len(train_df[train_df.LABEL == 1]))
      print('negative train trials:',len(train_df[train_df.LABEL == 0]))
      print('positive test trials:',len(test_df[test_df.LABEL == 1]))
      print('negative test trials:',len(test_df[test_df.LABEL == 0]))    
      

      train_X = train_df.drop(columns=['LABEL'])
      test_X = test_df.drop(columns=['LABEL'])
      
      train_y = train_df['LABEL'].astype(float)
      test_y = test_df['LABEL'].astype(float)
      
    
      # upsample training data with copies of data for the positive label
      if upsample:
        train_y_positive = train_y[train_y == 1]
        train_X_positive = train_X[train_X.index.isin(train_y_positive.index)]
        train_X = pd.concat([train_X,train_X_positive])
        train_y = pd.concat([train_y,train_y_positive])
    

      clf = LogisticRegression()
      clf.fit(train_X, train_y)
      preds = clf.predict(test_X)
      auc.append(aucScore(test_y,preds))


    # output is the average auc
    return sum(auc)/len(auc)




def regression_by_fixation(labels:dict,
                           max_fixations:int,
                           data_path:str,
                           results_path:str,
                           file_name:str,
                           positive_subjects:list,
                           negative_subjects:list,
                           phases:list =['Learning','Target','Distractor']):
  
    """
    Loading eyetracking data and transforming it to trials of length m fixations (for 1<m=<max_fixations) saving the results as a json file
    
    Args:
    
        labels: 
                dict with keys as positive_class and/or negative_class and values as file names for the positive anor negative classe
        
        max_fixations: 
                max number of fixations for which classification is performed
        
        data_path: 
                path where eyetracking data is saved
        
        results_path: 
                path where the results are saved,
        
        phases: 
                phases of the experiment to load data for
    
    """
    
    #load data
    data = EyeTrackingData(labels, data_path = data_path,phases=phases).Load() 
    
    
    # split subject data for cross validated training
    train_subjects, test_subjects = create_splits(
                                                  positive_subjects=positive_subjects,
                                                  negative_subjects=negative_subjects,
                                                  p_val_subjects=0)
    
    # train models for 1<fixations<=20
    results = {'num_fixations':[],'score':[]}
    
    for num_fixations in range(2,max_fixations+1):
        print(f'using {num_fixations} fixations (max={max_fixations})')
        
        trial_data = load_trials(data,num_fixations) # load trials (first n fixations of each trial)
        trial_data = preprocess(trial_data,num_fixations) # feature engineering 
        # record the fixation number and auc for the plot
        
        auc = train_classifier(trial_data, train_subjects, test_subjects,upsample=True)
        
        results['num_fixations'].append(num_fixations)
        results['score'].append(auc) 
    
    
    
    # save results
    with open(os.path.join(results_path,file_name), "w") as write_file:
        json.dump(results, write_file)



def plot_auc_vs_fixations(path, file_name):

    with open(os.path.join(path,file_name), "r") as read_file:
        results = json.load(read_file)
        
    fig = go.Figure(data=[go.Bar(x=results['num_fixations'], y=results['score'])])
    fig.update_xaxes(title='Number of Fixations',title_font_size=30,tickfont=dict(size=25))
    fig.update_yaxes(title='Performance (AUC)',title_font_size=30,tickfont=dict(size=25),range=[0.45, 0.67])
    fig.update_traces(marker_color='grey')
    fig.update_layout(paper_bgcolor='white',plot_bgcolor='white',width=1500)
    fig.show()