
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import random 
import statistics
from scipy import stats
from sklearn.decomposition import PCA
import os
import statistics
from sklearn.metrics import roc_auc_score as aucScore
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from PIL import Image
from sklearn.metrics import roc_auc_score as aucScore
from sklearn.metrics import classification_report
#from keras.optimizers import Adam
import warnings 
warnings.filterwarnings("ignore")
from tensorflow.keras.utils import array_to_img
from matplotlib.patches import Polygon
import plotly.graph_objects as go
from keras import callbacks
import matplotlib.image as mpimg
from matplotlib.patches import Polygon
from pylab import rcParams
import statistics
from matplotlib.colors import LinearSegmentedColormap
from keras import callbacks
from sklearn.metrics import roc_auc_score as aucScore
# path = "/content/drive/MyDrive/Colab Notebooks/Project 0/Project 0/Notebooks"
# import sys
# sys.path.append(path)
from analysis.analysis_tools.training import CreateSplits
from analysis.analysis_tools.processing import Image2Array




# visualizing group results
def TrialsByProbability(positive_class, path, labels, model_name, num_fixations, return_arrays = True, probability=0.1):

    image_folder = f'{model_name} - {positive_class} - {num_fixations}'
    positive_subjects =  os.listdir(os.path.join(path,'images',image_folder,'positive_class'))
    negative_subjects =  os.listdir(os.path.join(path,'images',image_folder,'negative_class'))
    
    _ ,_ , test_dicts = CreateSplits(positive_class,positive_subjects, negative_subjects, validation=True, return_list = False)
    
    positive = []
    negative = [] 
    positive_names = []
    positive_trials = []
    negative_names = []
    negative_trials = []

    for test_dict in test_dicts:
        fold = 0
        test_data = Image2Array(subjects=test_dict, path=path, labels=labels, image_folder= image_folder,num_fixations=num_fixations)
        test_X, test_y  = test_data.X, test_data.y
        names, trials = test_data.names, test_data.trials

        sum_preds = np.zeros((test_y.shape[0],1))
        for run_number in range(1,11):
            model = keras.models.load_model(os.path.join(path, 'models', positive_class, f'{model_name}_run_{run_number}_subject_{fold}'.lower()))
            preds = (model.predict(test_X)).reshape(test_y.shape[0],1)
            sum_preds += preds
        
        mean_preds = sum_preds/10
        
        for p in range(len(mean_preds)):
            if mean_preds[p] >= np.percentile(mean_preds, 100-(probability*100)):
                positive.append(test_X[p])
                positive_names.append(names[p])
                positive_trials.append(trials[p])
            elif mean_preds[p] <= np.percentile(mean_preds, probability*100):
                negative.append(test_X[p])  
                negative_names.append(names[p])
                negative_trials.append(trials[p])                
        fold += 1
    
    if return_arrays:
        return positive, negative
    else: 
        return positive_names, positive_trials, negative_names, negative_trials
 
 


def PlotFace(images,fig_path=None):
    
    plt.clf()

    if type(images[0]) == str: # if images = image paths 
        image_list = []
        for path in images:
            try:
                image_list.append(np.asarray(Image.open(path).convert('RGB')))
            except FileNotFoundError:
                pass
    
    else: # if images = image arrays 
        image_list = images
    
    
    image_arr = np.zeros((image_list[0].shape[0],image_list[0].shape[1],3))
    for i in image_list:
        image_arr += i
    #image_arr = image_arr ** 1/12
    
    
    
    fig, axis = plt.subplots(nrows=1, ncols=1,figsize=(15,15))
    axis.matshow(image_arr[:,:,0],cmap='inferno')
    return

    
    
    
    
    
def PlotROI(images,num_fixations,fig_path=None):
    

    plt.clf()
    
    if type(images[0]) == str: # if images = image paths 
        image_list = []
        for path in images:
            try:
                image_list.append(np.asarray(Image.open(path).convert('RGB')))
            except FileNotFoundError:
                pass
    
    else: # if images = image arrays 
        image_list = images
    
    
    image_arr = np.zeros((image_list[0].shape[0],image_list[0].shape[1],3))
    for i in image_list:
        image_arr += i
    #image_arr = image_arr ** 1/4
    
    
    fig, axis = plt.subplots(nrows=1, ncols=1,figsize=(15,15))
    xTicksRange = np.arange(image_list[0].shape[1]/20,image_list[0].shape[1],image_list[0].shape[1]/10)
    xTickLabels = ['Forehead','LeftEyebrow','RightEyebrow', 'LeftEye', 'RightEye', 'LeftCheek', 'RightCheek', 'Nose', 'Mouth','Chin']
    yTicksRange = np.arange(image_list[0].shape[0]/(num_fixations*2),image_list[0].shape[0],image_list[0].shape[0]/num_fixations)
    yTickLabels = [f'Fixation {i}' for i in range(1,num_fixations+1)]

    axis.imshow(array_to_img(image_arr))
    axis.set_xticks(ticks = xTicksRange)
    axis.set_yticks(ticks = yTicksRange)
    axis.set_xticklabels(xTickLabels,rotation=35,fontsize=25)
    axis.set_yticklabels(yTickLabels,fontsize=22)      
    
    return
