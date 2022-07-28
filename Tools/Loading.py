
import pandas as pd
import numpy as np
import random 
from scipy import stats
import os
import statistics
import statistics


def LoadDataFromFile(labels,phases,data_path):
    
    df_list = []
    
    for group, file_name in labels.items():
      assert group in ['Positive_Class','Negative_Class']
      tmp = []
      
      for phase in phases:
        phase_df = pd.read_excel(f'{data_path}/{file_name}.xlsx', f'{phase}')
        phase_df['PHASE'] = phase
        phase_df['LABEL'] = 0
        if group == 'Positive_Class':
            phase_df['LABEL'] = 1
        tmp.append(phase_df)
    
      df_list.append(pd.concat(tmp))
    
    return pd.concat(df_list)
    

# Importing Eyetracker Data for all Phases of the Experiment 
class EyeTrackingData:
    def __init__(self,data_path,
                labels={'Positive_Class':'AP - New ROI Data','Negative_Class':'Control - New ROI Data'},
                phases=['Learning','Target','Distractor']):
        
        """ 
        data_path: path to the eyetracking data
        labels: dictionary of file names decsribing the positive class and negative class
        phases: the phases of the experiment to load
        """
        
        self.data_path = data_path
        self.labels = labels
        self.phases = phases

    def Load(self):
    
        df = LoadDataFromFile(self.labels,self.phases,self.data_path)
        df['TRIAL_INFO'] = df.RECORDING_SESSION_LABEL+ df.TRIAL_LABEL
        return df
    

    def Names(self):
        
        df = LoadDataFromFile(self.labels,self.phases,self.data_path)
        return df.RECORDING_SESSION_LABEL.unique().to_list()
  




# Format data, process columns and shrink dataset based on the number of fixations taken as input. Outputs a list of trials, each a separate dataframe
def LoadTrials(df,number_of_fixations):
  
  roi_dict = {'DURATION_CENTRAL':['LeftEye','RightEye','Nose','Mouth'],
            'DURATION_PERIPHERAL':['LeftEye','RightEye','Nose','Mouth'],
            'DURATION_RIGHT': ['RightEyebrow', 'RightEye','RightCheek'],
            'DURATION_LEFT':['LeftEyebrow','LeftEye','LeftCheek'],
            'DURATION_EYES':['LeftEye','RightEye'],
            'DURATION_NOSE_MOUTH':['Nose','Mouth'],
            'DURATION_EYEBROWS':['LeftEyebrow','RightEyebrow'],
            'DURATION_FOREHEAD':['Forehead'],
            'DURATION_CHEEKS':['RightCheek','LeftCheek'],
            'DURATION_CHIN':['Chin']
            }

  allTrials = []

  for i in list(df.TRIAL_INFO.unique()):
      trial = df[df.TRIAL_INFO == i]  
      
      if len(trial) >= number_of_fixations:

          trial['DURATION_TOTAL'] = trial.CURRENT_FIX_DURATION.sum()
          trial = trial.iloc[:number_of_fixations,:]
          trial[f'DURATION_FIRST_{number_of_fixations}'] = trial.CURRENT_FIX_DURATION.sum()
          
          ROIS = ['LeftEyebrow', 'RightEyebrow', 'LeftEye', 'RightEye', 'Forehead',
                    'Nose', 'Mouth', 'Chin', 'LeftCheek', 'RightCheek']

          for roi in ROIS:
              trial[roi] = trial[roi].map({'Yes':1,'No':0})

          trial = trial.reset_index(drop=True)
          x= trial[ROIS].stack()
          
          trial['CURRENT_FIX_INDEX'] = pd.DataFrame(pd.Categorical(x[x!=0].index.get_level_values(1)))
          trial['DURATION_MEAN'] = trial['CURRENT_FIX_DURATION'].mean()
          trial['DURATION_VARIANCE'] = statistics.variance(trial['CURRENT_FIX_DURATION'])
            
          for key, values in roi_dict.items():
            trial[key] = trial[['CURRENT_FIX_INDEX','CURRENT_FIX_DURATION']].apply(lambda x: 0 if x.CURRENT_FIX_INDEX in values else x.CURRENT_FIX_DURATION,axis=1) 

          trial['PHASE'] = trial['PHASE'].map({'Distractor':0,'Target':1,'Learning':2})

          allTrials.append(trial)
  
  return allTrials

