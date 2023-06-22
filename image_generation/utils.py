
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import os
from matplotlib.patches import Polygon
from pylab import rcParams
from matplotlib.colors import LinearSegmentedColormap
from tools.loading import *
from tools.processing import *





def load_data(data_path,num_fixations,labels,phases):
    """
    Load eyetracking data and extact trials with their first m fixations where m = num_fixations  
    
    Args
        data_path: path to the eyetracking data
        num_fixations: number of fixations to include from both classes
        labels: dict with keys as positive_class and/or negative_class and values as file names for the positive anor negative classe
        phases: the phases of the experiment to load
    
    Output
        datafarme of trials with their first m fixations where m = num_fixations  
    """

    eyetracking = EyeTrackingData(data_path = data_path,labels=labels,phases=phases)
    data = eyetracking.Load()
    trials = load_trials(data,num_fixations)
    return pd.concat(trials)





def make_dirs(num_fixations, image_type, image_path, labels):
    
    """
    Create directories for all the folders if dr doesnt exist 
    
    Args:
        num_fixations: number of fixations used to generate the images
        image_type: type of image generation algorithm (temporal or scanpath) 
        image_path: path to where images will be saved
        positive_class: name of the positive class for classification will be done
        
    Output:
        name of the folder where images will be saved
    
    """
    folder_name = image_type + ' - ' + labels['positive_class'] + ' - ' + str(num_fixations)
    
    if not os.path.exists(image_path):
        os.mkdir(image_path)
        
    if not os.path.exists(os.path.join(image_path,folder_name)):
        os.mkdir(os.path.join(image_path,folder_name))
    
    return folder_name







#gerenating scanpath images
def make_scanpath_images(data_path:str,
                    image_path:str,
                    num_fixations:int,
                    labels:dict,
                    phases=['Learning','Target','Distractor']):
                        
    """ 
    Generate and save images using the scanpath generation algorithm
    
    Args:
        data_path: path to the eyetracking data
        image_path: path where images will be saved 
        num_fixations: number of fixations to include from both classes
        labels: dict with keys as positive_class and/or negative_class and values as file names for the positive anor negative classe
        phases: the phases of the experiment to load
    """
    
    plt.style.use('dark_background')
    rcParams['figure.figsize'] = 5, 5
    cmap = LinearSegmentedColormap.from_list('name', ['white', 'dimgray'])
    
    folder_name = make_dirs(num_fixations=num_fixations, image_type='scanpath', image_path=image_path, labels=labels)
    


    for group,file_name in labels.items():
        if not os.path.exists(os.path.join(image_path,folder_name,group)):
            os.mkdir(os.path.join(image_path,folder_name,group))
        label = {group:file_name}
        df = load_data(data_path,num_fixations,label,phases)
        
        # scale duration sizes
        df['CURRENT_FIX_DURATION_SQUARED'] = (df['CURRENT_FIX_DURATION']**2)/400   
        
        for subject in df.RECORDING_SESSION_LABEL.unique():
            if not os.path.exists(os.path.join(image_path, folder_name, group, subject)):
                os.mkdir(os.path.join(image_path,folder_name,group, subject))
            subject_trials = df[df.RECORDING_SESSION_LABEL == subject]
        
    
            for trial in subject_trials.TRIAL_LABEL.unique():
              trial_data = subject_trials[subject_trials.TRIAL_LABEL == trial]
              trial_num = trial_data.TRIAL_LABEL.unique()[0]
              
              plt.clf()
              plt.xlim(345,675)
              plt.ylim(600,150)
              
              # face outline
              pts = np.array([[372,201], [649,201], [655,422],[607,530],[511,576],[416,530],[364,422]])
              p = Polygon(pts,ec='white',fc='none')
              ax = plt.gca()
              ax.add_patch(p)
              
              # fixations
              plt.scatter(trial_data.CURRENT_FIX_X,trial_data.CURRENT_FIX_Y,edgecolor='none',s=trial_data.CURRENT_FIX_DURATION_SQUARED,c=trial_data.index,cmap=cmap)
              plt.axis('off')
              plt.savefig(os.path.join(image_path,folder_name,group,subject,f'{trial_num}.png'),bbox_inches = 'tight',pad_inches=0)
      
      
      
      
      
      
# generating ROI sequence images
def make_temporal_images(data_path:str,
                    image_path:str,
                    num_fixations:int,
                    labels:dict,
                    phases=['Learning','Target','Distractor']):
    
    """ 
    Generate and save images using the temporal generation algorithm
    
    Args:
        data_path: path to the eyetracking data
        image_path: path where images will be saved 
        num_fixations: number of fixations to include from both classes
        labels: dict with keys as positive_class and/or negative_class and values as file names for the positive anor negative classe
        phases: the phases of the experiment to load
    """
    
    plt.style.use('dark_background')
    rcParams['figure.figsize'] = 5, 5
    cmap = LinearSegmentedColormap.from_list('name', ['black','white'])
    
    folder_name = make_dirs(num_fixations=num_fixations, image_type='temporal', image_path=image_path,labels=labels)
    rois= ['Forehead','LeftEyebrow','RightEyebrow', 'LeftEye', 'RightEye', 'LeftCheek', 'RightCheek', 'Nose', 'Mouth','Chin']

    for group,file_name in labels.items():
        if not os.path.exists(os.path.join(image_path,folder_name,group)):
            os.mkdir(os.path.join(image_path,folder_name,group))
        label = {group:file_name}
        df = load_data(data_path,num_fixations,label,phases)
        # scale duration sizes
        df['CURRENT_FIX_DURATION_SQUARED'] = (df['CURRENT_FIX_DURATION']**2)/400   
        
        for subject in df.RECORDING_SESSION_LABEL.unique():
            if not os.path.exists(os.path.join(image_path, folder_name, group, subject)):
                os.mkdir(os.path.join(image_path,folder_name,group, subject))
            subject_trials = df[df.RECORDING_SESSION_LABEL == subject]
        
    
            for trial in subject_trials.TRIAL_LABEL.unique():
              trial_data = subject_trials[subject_trials.TRIAL_LABEL == trial]
              trial_num = trial_data.TRIAL_LABEL.unique()[0]
 
              arr = np.array(trial_data[rois])[0]
              for i in range(1,num_fixations):
                arr = np.vstack((arr,np.array(trial_data[rois])[i]))
    
              plt.clf()
              sns.heatmap(arr,cbar=False,xticklabels=False,yticklabels=False,cmap=cmap)
              plt.margins(0)
              plt.axis('off')
              plt.savefig(os.path.join(image_path,folder_name,group, subject,f'{trial_num}.png'),bbox_inches = 'tight',pad_inches=0)
              



    