
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
path = "/content/drive/MyDrive/Colab Notebooks/Project 0/Project 0/Notebooks"
import sys
sys.path.append(path)
from Analysis.Tools.Training import *




# CNN Model Architecture 
def Model(lr,input_shape,convlayer,convdim):
    
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
def TrainCNN(model,train_X,train_y,val_X,val_y,batch_size,patience):
  earlystopping = callbacks.EarlyStopping(monitor ="val_accuracy", mode ="max", patience = patience, restore_best_weights = True)
  model.fit(train_X, train_y, steps_per_epoch=len(train_X) // batch_size,batch_size = batch_size, epochs=30,validation_data=(val_X,val_y),callbacks =[earlystopping])
  return model
  
  



# Different Evaluation Metrics
def Evaluate(model,X,y,metric_type):
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





# Plotting Loss and Accuracy
def PlotLoss(hist):
    
  fig, (ax1, ax2) = plt.subplots(1, 2)
  ax1.plot(hist.history['loss'], label='train')
  ax1.plot(hist.history['val_loss'], label='validation')
  ax1.set_title('Loss')
  ax1.legend()

  ax2.plot(hist.history['accuracy'], label='train')
  ax2.plot(hist.history['val_accuracy'], label='validation')
  ax2.set_title('Accuracy')
  ax2.legend()





# Getting the CNN Model Results using the Validation Set
def ValResults(image_folder,image_path, model_path, model_name, labels, positive_class,patience):
      
  results = {'train_set':[],'val_set':[],'test_set':[],'train_acc':[],'train_auc':[],'val_acc':[],'val_auc':[]} 
  train, val, test = TrainTestSplit(image_path,positive_class=positive_class,validation=True,image_folder=image_folder)

  for i in range(len(test)):

      results['train_set'].append(train[i])
      results['val_set'].append(val[i])
      results['test_set'].append(test[i])

      train_data = Image2Array(train[i], image_path, 'RGB', image_folder)
      train_data.upsample()
      train_data.shuffle()
      train_X, train_y  = train_data.X, train_data.y 

      val_data = Image2Array(val[i], image_path, 'RGB', image_folder)
      val_X, val_y  = val_data.X, val_data.y 

      model = Model(lr=0.0001,input_shape=train_X.shape[1:],convlayer=3,convdim=[32,64,128]) 
      model = TrainCNN(model,train_X, train_y, val_X, val_y, batch_size=32,patience=patience)
      

      results['train_accuracy'].append(Evaluate(model,train_X, train_y,'accuracy'))
      results['train_accuracy'].append(Evaluate(model,train_X, train_y,'auc'))
      print('train_accuracy',Evaluate(model,train_X, train_y,'accuracy'),'train_accuracy',Evaluate(model,train_X, train_y,'auc'))

      results['val_accuracy'].append(Evaluate(model,valX, valy,'accuracy'))
      results['val_accuracy'].append(Evaluate(model,valX, valy,'auc'))
      print('val_accuracy',Evaluate(model,valX, valy,'accuracy'),'val_accuracy',getMetrics(model,valX, valy,'auc'))

      model.save(f'{model_path}/{model_name}_{i}')

  return pd.DataFrame(results)





# Getting the CNN Model Results using the Test Set
def TestResults(image_folder,image_path,model_path,model_name,labels,positive_class):
      
  results = {'test_accuracy':[],'test_auc':[],'negative_class_ratio':[],'positive_class_ratio':[]} 
  _, _, test = TrainTestSplit(image_path,positive_class=positive_class,val_set=True,image_folder=image_folder)

  for i in range(len(test)):

    test_data = Image2Array(test[i], image_path, 'RGB', image_folder)
    test_X, test_y  = test_data.X, test_data.y

    results['negative_class_ratio'].append(len(np.where(testy == 0)[0])/len(test_y))
    results['positive_class_ratio'].append(len(np.where(testy == 1)[0])/len(test_y))

    model = keras.models.load_model(f'{model_path}/{model_name}_{i}')
    preds = model.predict(test_X)  
    
    results['test_accuracy'].append(Evaluate(model,test_X, test_y,'accuracy'))
    results['test_auc'].append(Evaluate(model,test_X, test_y,'auc'))

  return pd.DataFrame(results)
  
  
 
 
  
# training random hyrbid models 
def RandomHybridModel(path,image_folder,positive_class,num_fixations,patience):
  train, val, test = TrainTestSplit(path,positive_class=positive_class,validation=True,image_folder=image_folder)
  auc = []
  
  for i in range(len(test)): 
      # Getting Random Scanpath Predictions for ith Cross Validation step
      trainData = Image2Array('RGB', train[i], path, f'Scanpath Images - {positive_class} - {num_fixations}')
      trainData.upsample()
      trainData.shuffle()
      trainX, trainy  = trainData.X, trainData.y 
      np.random.shuffle(trainy)

      valData = Image2Array('RGB', val[i], path, f'Scanpath Images - {positive_class} - {num_fixations}')
      valX, valy  = valData.X, valData.y 
      
      testData = Image2Array('RGB', test[i], path,f'Scanpath Images - {positive_class} - {num_fixations}')
      testX, testy  = testData.X, testData.y 
      
      scanModel = Model(lr=0.0001,input_shape=trainX.shape[1:],convlayer=3,convdim=[32,64,128]) 
      scanModel = TrainCNN(scanModel,trainX, trainy, valX, valy, batch_size=32,patience=patience)
      scanPreds = scanModel.predict(testX)     

      # Getting Random Temporal Predictions for ith Cross Validation step
      trainData = Image2Array('RGB', train[i], path, f'Temporal Images - {positive_class} - {num_fixations}')
      trainData.upsample()
      trainData.shuffle()
      trainX, trainy  = trainData.X, trainData.y 
      np.random.shuffle(trainy)

      valData = Image2Array('RGB', val[i], path, f'Temporal Images - {positive_class} - {num_fixations}')
      valX, valy  = valData.X, valData.y 
      
      testData = Image2Array('RGB', test[i], path, f'Temporal Images - {positive_class} - {num_fixations}')
      testX, testy  = testData.X, testData.y 
    

      tempModel = Model(lr=0.0001,input_shape=trainX.shape[1:],convlayer=3,convdim=[32,64,128]) 
      tempModel = TrainCNN(tempModel,trainX, trainy, valX, valy, batch_size=32,patience=patience)
      tempPreds = tempModel.predict(testX)
      

      aucList = []
      weights = np.linspace(0,1,1000)
      for w in weights:
        iw = 1-w
        aucList.append(aucScore(testy,(w * scanPreds+ iw * tempPreds))) 

      auc.append(max(aucList))
      
  return (sum(auc)/len(auc))