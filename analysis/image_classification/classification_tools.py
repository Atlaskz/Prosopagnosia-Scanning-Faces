
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import random 
import statistics
from scipy import stats
import os
from sklearn.metrics import roc_auc_score as aucScore
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
from keras import callbacks
import statistics
from sklearn.metrics import roc_auc_score as AUCScore
import sys
sys.path.append("/content/drive/MyDrive/Colab Notebooks/Project 0/Project 0/Notebooks")
from analysis.analysis_tools.training import *
from analysis.analysis_tools.processing import *
import tensorflow as tf


# CNN Model Architecture 
def Model(lr:float,input_shape:tuple,convlayer:int,convdim:list):
    
  """
  Initializes the CNN model
  
  Args:
      lr: learning rate 
      input_shape: shape of the input image
      convlayer: number of convolution layers
      convdim: the number of output filters in the convolution.length of list muct match number of conv layers
  
  Output:
    the CNN model 
  
  """
  classifier = Sequential()
  for layer in range(convlayer):
    if layer == 0:
      classifier.add(Convolution2D(convdim[layer], 3, 3, input_shape = input_shape, activation = 'relu'))
      classifier.add(MaxPooling2D(pool_size = (2,2)))
    else:
      classifier.add(Convolution2D(convdim[layer], 3, 3, activation = 'relu'))
      classifier.add(MaxPooling2D(pool_size = (2,2)))
  
  classifier.add(Flatten())
  classifier.add(Dense(64, activation = 'tanh'))
  classifier.add(Dropout(0.5))
  classifier.add(Dense(16, activation = 'tanh'))
  classifier.add(Dense(1, activation = 'sigmoid'))
  classifier.compile(optimizer = keras.optimizers.Adam(lr), loss = 'binary_crossentropy', metrics = ['accuracy'])
  return classifier





# Training the CNN Model
def FitModel(model,train_X,train_y,val_X,val_y,batch_size,patience):
  
  earlystopping = callbacks.EarlyStopping(monitor ="val_accuracy", mode ="max", patience = patience, restore_best_weights = True)
  model.fit(train_X, train_y, steps_per_epoch=len(train_X) // batch_size,batch_size = batch_size, epochs=30,validation_data=(val_X,val_y),callbacks =[earlystopping])
  return model
  
  



# Different Evaluation Metrics
def Evaluate(model,X:np.array,y:np.array,metric_type:str):
    """
    Evaluates the results of the model using different metrics
    
    Args:
        model: the trained CNN model
        X: data to test 
        y: the true label values
        metric_type: type of metric used (accuracy, auc or recall)

    Output:
        evaluation results
      
    """
    if metric_type == 'accuracy':
        return model.evaluate(np.array(X),y)[1]
    
    elif metric_type == 'auc':
        preds = model.predict(np.array(X))
        return AUCScore(y,preds)
    
    elif metric_type == 'recall':
        target_names = ['negative', 'positive']
        preds = model.predict(np.array(X))
        report = classification_report(y, preds, target_names=target_names, output_dict=True)
        return report['negative']['recall'], report['positive']['recall']





# Getting the CNN Model Results using the Validation Set
def Train(path:str, image_folder:str, model_name:str, num_fixations:int, labels:dict, patience:int =10):
      
    '''
    Trains the CNN model using a k-fold cross validation design and saves the trained models
    
    Args:
        path: root directory path
        image_folder: folder to get the images from
        model_name: name of the model (scanpath or temporal)
        num_fixations: the number of fixations used for generating the images
        labels: dictionary of the form {positive_class : *name of positive class*, negative_class : *name of negative class*} 
        patience: Number of epochs with no improvement after which training will be stopped (https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping).
 
    '''
    
    
    results = {'train_set':[],'val_set':[],'test_set':[],'train_accuracy':[],'train_auc':[],'val_accuracy':[],'val_auc':[]} 
    
    positive_subjects = os.listdir(os.path.join(path,'images',image_folder,'positive_class'))
    negative_subjects = os.listdir(os.path.join(path,'images',image_folder,'negative_class'))
    
    train, val, test = CreateSplits(labels['positive_class'], positive_subjects, negative_subjects,validation=True,return_list=False)
    
    for i in range(len(test)):
        
        results['train_set'].append(train[i])
        results['val_set'].append(val[i])
        results['test_set'].append(test[i])
        
        
        train_data = Image2Array(subjects=train[i], path=path, labels=labels, image_folder = image_folder ,num_fixations = num_fixations)
        train_data.upsample()
        train_data.shuffle()
        train_X, train_y  = train_data.X, train_data.y 
        
        val_data = Image2Array(val[i], path=path, labels=labels, image_folder = image_folder ,num_fixations = num_fixations)
        val_X, val_y  = val_data.X, val_data.y 
        
        for run_number in range(1,11):
            print(f'Run {run_number} out of 10')
            
            model = Model(lr=0.0001,input_shape=train_X.shape[1:],convlayer=3,convdim=[32,64,128]) 
            model = FitModel(model,train_X, train_y, val_X, val_y, batch_size=32,patience=patience)
        
            results['train_accuracy'].append(Evaluate(model,train_X, train_y,'accuracy'))
            results['train_auc'].append(Evaluate(model,train_X, train_y,'auc'))
            print('train_accuracy',Evaluate(model,train_X, train_y,'accuracy'),'train_accuracy',Evaluate(model,train_X, train_y,'auc'))
        
            results['val_accuracy'].append(Evaluate(model,val_X, val_y,'accuracy'))
            results['val_auc'].append(Evaluate(model,val_X, val_y,'auc'))
            print('val_accuracy',Evaluate(model,val_X, val_y,'accuracy'),'val_auc',Evaluate(model,val_X, val_y,'auc'))
        
            model.save(os.path.join(path,'models', labels['positive_class'], f'{model_name}_run_{run_number}_subject_{i}'))





# Getting the CNN Model Results using the Test Set
def Test(path:str, image_folder:str, model_name:str, num_fixations:int, labels:dict):
    
    '''
    Tests the trained models on held out data
    
    Args:
        path: path to the root directory
        image_folder: name of folder where images are saved
        model_name: name of the model (temporal or scanpath)
        num_fixations: the number of fixations used to generate images
        labels: dictionary of the form {positive_class : *name of positive class*, negative_class : *name of negative class*} 
        
    Output:
        Dataframe with the results of k-fold corss validation on test data
    
    '''
    positive_subjects = os.listdir(os.path.join(path,'images',image_folder,'positive_class'))
    negative_subjects = os.listdir(os.path.join(path,'images',image_folder,'negative_class'))
          
    all_results = pd.DataFrame.from_dict({'test_accuracy':[],'test_auc':[],'negative_class_ratio':[],'positive_class_ratio':[]})
    _, _, test = CreateSplits(labels['positive_class'], positive_subjects, negative_subjects,validation=True, return_list = False)
    
    
    k = 1
    for i in range(len(test)):
    
    
        test_data = Image2Array(subjects=test[i], path=path, labels=labels, image_folder = image_folder ,num_fixations = num_fixations)
        test_X, test_y  = test_data.X, test_data.y
        
        for run_number in range(1,11):
            
            results = {'test_accuracy':[],'test_auc':[],'negative_class_ratio':[],'positive_class_ratio':[],'fold':[]} 
            
            results['negative_class_ratio'].append(len(np.where(test_y == 0)[0])/len(test_y))
            results['positive_class_ratio'].append(len(np.where(test_y == 1)[0])/len(test_y))
        
            model = keras.models.load_model(os.path.join(path,'models', labels['positive_class'], f'{model_name}_run_{run_number}_subject_{i}'))
            preds = model.predict(test_X)  
            
            results['test_accuracy'].append(Evaluate(model,test_X, test_y,'accuracy'))
            results['test_auc'].append(Evaluate(model,test_X, test_y,'auc'))
            results['fold'].append(k)
            
            results_df = pd.DataFrame.from_dict(results)
            all_results = pd.concat([all_results,results_df])
        k+=1
    pc = labels['positive_class']
    nc = labels['negative_class']
    all_results.to_csv(os.path.join(path,'results',f"results_{pc}_vs_{ng}"))
        
    
    
    
    
