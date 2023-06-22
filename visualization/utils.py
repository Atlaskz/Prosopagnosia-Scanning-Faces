import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from PIL import Image
import warnings 
warnings.filterwarnings("ignore")
from tensorflow.keras.utils import array_to_img

from tools.training import load_splits
from tools.processing import Image2Array



# visualizing group results
def trials_by_probability(splits_path, positive_class, path, labels, model_name, num_fixations, return_arrays = True, probability=0.1):

    image_folder = f'{model_name} - {positive_class} - {num_fixations}'

    _, _, test_dicts = load_splits(splits_path)

    positive = []
    negative = [] 
    positive_names = []
    positive_trials = []
    negative_names = []
    negative_trials = []

    print(test_dicts)
    for test_dict in test_dicts:
        fold = 0
        test_data = Image2Array(subjects=test_dict, path=path, labels=labels, image_folder= image_folder,num_fixations=num_fixations)
        test_X, test_y  = test_data.X, test_data.y
        names, trials = test_data.names, test_data.trials
        print(names)
        print(trials)

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
 
 


def plot_face(images,factor=1,fig_path=None):
    
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
    image_arr = image_arr ** factor
    
    
    
    fig, axis = plt.subplots(nrows=1, ncols=1,figsize=(5,5))
    axis.matshow(image_arr[:,:,0],cmap='inferno')
    fig.savefig(fig_path)
    return

    
    
    
    
    
def plot_roi(images,num_fixations,fig_path=None):
    

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
    
    
    fig, axis = plt.subplots(nrows=1, ncols=1,figsize=(5,5))
    xTicksRange = np.arange(image_list[0].shape[1]/20,image_list[0].shape[1],image_list[0].shape[1]/10)
    xTickLabels = ['Forehead','LeftEyebrow','RightEyebrow', 'LeftEye', 'RightEye', 'LeftCheek', 'RightCheek', 'Nose', 'Mouth','Chin']
    yTicksRange = np.arange(image_list[0].shape[0]/(num_fixations*2),image_list[0].shape[0],image_list[0].shape[0]/num_fixations)
    yTickLabels = [f'Fixation {i}' for i in range(1,num_fixations+1)]

    axis.imshow(array_to_img(image_arr))
    axis.set_xticks(ticks = xTicksRange)
    axis.set_yticks(ticks = yTicksRange)
    axis.set_xticklabels(xTickLabels,rotation=35,fontsize=10)
    axis.set_yticklabels(yTickLabels,fontsize=10)      
    fig.savefig(fig_path)
    
    return