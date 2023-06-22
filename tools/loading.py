
import pandas as pd
import statistics


def load_data_from_file(labels,phases,data_path):
    
    """
    Load eyetracking data
    
    Args:
        labels: dict with keys as positive_class and/or negative_class and values as file names for the positive anor negative classe
        phases: phases of the experiment to load
        data_path: path where the data exists
    Output:
        dataframe containing eyetracking data for the positive and negative class
    
    """
    
    df_list = []
    
    for label, file_name in labels.items():
      tmp = []
      
      for phase in phases:
        phase_df = pd.read_excel(f'{data_path}/{file_name}.xlsx', f'{phase}')
        phase_df['PHASE'] = phase
        phase_df['LABEL'] = 0
        if label == 'positive_class':
            phase_df['LABEL'] = 1
        tmp.append(phase_df)
    
      df_list.append(pd.concat(tmp))
    
    return pd.concat(df_list)
    

# Importing Eyetracker Data for all Phases of the Experiment 
class EyeTrackingData:
    def __init__(self,labels,data_path,
                phases=['Learning','Target','Distractor']):
        
        """ 
        Args:
            labels: dict with keys as positive_class and/or negative_class and values as file names for the positive anor negative classe
            data_path: path to the eyetracking data
            phases: the phases of the experiment to load
        
        """
        
        self.labels = labels
        self.data_path = data_path
        self.phases = phases

    def Load(self):
        """Load eyetracking data from data path
        """
    
        df = load_data_from_file(self.labels,self.phases,self.data_path)
        df['TRIAL_INFO'] = df.RECORDING_SESSION_LABEL+ df.TRIAL_LABEL
        return df
    

    def Names(self):
        """
        Load subject names from eyetracking data
        """
        df = load_data_from_file(self.labels,self.phases,self.data_path)
        return list(df.RECORDING_SESSION_LABEL.unique())
  




# Format data, process columns and shrink dataset based on the number of fixations taken as input. Outputs a list of trials, each a separate dataframe
def load_trials(df,number_of_fixations):
  
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

