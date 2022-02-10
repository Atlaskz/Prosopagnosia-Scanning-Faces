
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
from keras.preprocessing.image import array_to_img
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





# Importing Eyetracker Data for all Phases of the Experiment 
def getData(path,labels=['AP - New ROI Data','Control - New ROI Data'],phases=['Learning','Target','Distractor'],subjectNames=False):

  allData = []
  
  for l in labels:
      tmp = []
      
      for p in phases:
        phase = pd.read_excel(f'{path}/{l}.xlsx', f'{p}')
        phase['PHASE'] = p
        phase['LABEL'] = l.split(' -')[0]
        tmp.append(phase)
        
      phasesDf = pd.concat(tmp)
      allData.append(phasesDf)

  df = pd.concat(allData)
  df['TRIAL_INFO'] = df.RECORDING_SESSION_LABEL	+ df.TRIAL_LABEL
  names = sorted(df.RECORDING_SESSION_LABEL.unique().tolist())

  if subjectNames==False:
    return df
  else:
    return names

# Format data, process columns and shrink dataset based on the number of fixations taken as input. Outputs a list of trials, each a separate dataframe
def getTrialData(data,number_of_fixations,positive='AP'):
  
  allTrials = []

  for i in list(data.TRIAL_INFO.unique()):
      trial = data[data.TRIAL_INFO == i]  
      if len(trial) >= number_of_fixations:

          trial['DURATION_TOTAL'] = trial.CURRENT_FIX_DURATION.sum()

          trial = trial.iloc[:number_of_fixations,:]

          trial[f'DURATION_FIRST_{number_of_fixations}'] = trial.CURRENT_FIX_DURATION.sum()

          ROIS = ['LeftEyebrow', 'RightEyebrow', 'LeftEye', 'RightEye', 'Forehead',
                    'Nose', 'Mouth', 'Chin', 'LeftCheek', 'RightCheek']

          for roi in ROIS:
              trial[roi] = trial[roi].map({'Yes':1,'No':0})

          trial = trial.reset_index(drop=True)
          x= trial[['LeftEyebrow', 'RightEyebrow', 'LeftEye', 'RightEye', 'Forehead',
                    'Nose', 'Mouth', 'Chin', 'LeftCheek', 'RightCheek']].stack()
          trial['CURRENT_FIX_INDEX'] = pd.DataFrame(pd.Categorical(x[x!=0].index.get_level_values(1)))

          trial['DURATION_CENTRAL'] = trial[['CURRENT_FIX_INDEX','CURRENT_FIX_DURATION']].apply(lambda x: x.CURRENT_FIX_DURATION if x.CURRENT_FIX_INDEX in ['LeftEye','RightEye','Nose','Mouth'] else 0,axis=1) 
          trial['DURATION_PERIPHERAL'] = trial[['CURRENT_FIX_INDEX','CURRENT_FIX_DURATION']].apply(lambda x: 0 if x.CURRENT_FIX_INDEX in ['LeftEye','RightEye','Nose','Mouth'] else x.CURRENT_FIX_DURATION,axis=1) 
          trial['DURATION_MEAN'] = trial['CURRENT_FIX_DURATION'].mean()
          trial['DURATION_VARIANCE'] = statistics.variance(trial['CURRENT_FIX_DURATION'])
          trial['DURATION_RIGHT'] = trial[['CURRENT_FIX_INDEX','CURRENT_FIX_DURATION']].apply(lambda x: x.CURRENT_FIX_DURATION if x.CURRENT_FIX_INDEX in ['RightEyebrow', 'RightEye','RightCheek',] else 0,axis=1)
          trial['DURATION_LEFT'] = trial[['CURRENT_FIX_INDEX','CURRENT_FIX_DURATION']].apply(lambda x: x.CURRENT_FIX_DURATION if x.CURRENT_FIX_INDEX in ['LeftEyebrow','LeftEye','LeftCheek'] else 0,axis=1)
          trial['DURATION_EYES'] = trial[['CURRENT_FIX_INDEX','CURRENT_FIX_DURATION']].apply(lambda x: x.CURRENT_FIX_DURATION if x.CURRENT_FIX_INDEX in ['LeftEye','RightEye'] else 0,axis=1)
          trial['DURATION_NOSE_MOUTH'] = trial[['CURRENT_FIX_INDEX','CURRENT_FIX_DURATION']].apply(lambda x: x.CURRENT_FIX_DURATION if x.CURRENT_FIX_INDEX in ['Nose','Mouth'] else 0,axis=1) 
          trial['DURATION_EYEBROWS'] = trial[['CURRENT_FIX_INDEX','CURRENT_FIX_DURATION']].apply(lambda x: x.CURRENT_FIX_DURATION if x.CURRENT_FIX_INDEX in ['LeftEyebrow','RightEyebrow'] else 0,axis=1)
          trial['DURATION_FOREHEAD'] = trial[['CURRENT_FIX_INDEX','CURRENT_FIX_DURATION']].apply(lambda x: x.CURRENT_FIX_DURATION if x.CURRENT_FIX_INDEX in ['Forehead'] else 0,axis=1)
          trial['DURATION_CHEEKS'] = trial[['CURRENT_FIX_INDEX','CURRENT_FIX_DURATION']].apply(lambda x: x.CURRENT_FIX_DURATION if x.CURRENT_FIX_INDEX in ['RightCheek','LeftCheek'] else 0,axis=1)
          trial['DURATION_CHIN'] = trial[['CURRENT_FIX_INDEX','CURRENT_FIX_DURATION']].apply(lambda x: x.CURRENT_FIX_DURATION if x.CURRENT_FIX_INDEX in ['Chin'] else 0,axis=1)

          trial['PHASE'] = trial['PHASE'].map({'Distractor':0,'Target':1,'Learning':2})
          trial['LABEL'] = trial['LABEL'].map({f'{positive}':1,'Control':0})

          allTrials.append(trial)
  return allTrials


# Processing the List of trials to produce a DataFrame of scanpaths with each scanpath represented as a single row 
def processData(allTrials,number_of_fixations):

  roi_dic = {'LeftEyebrow':1, 'RightEyebrow':2, 'LeftEye':3, 'RightEye':4, 'Forehead':5, 'Nose':6, 'Mouth':7, 'Chin':8, 'LeftCheek':9, 'RightCheek':10}
  
  cols =['RECORDING_SESSION_LABEL','CURRENT_FIX_INDEX',f'DURATION_FIRST_{number_of_fixations}', 'DURATION_TOTAL','DURATION_MEAN','DURATION_VARIANCE',
         'PHASE','LABEL','DURATION_FOREHEAD','DURATION_EYEBROWS','DURATION_CHEEKS','DURATION_CHIN']
  
  roi_cols = ['DURATION_EYES','DURATION_NOSE_MOUTH','DURATION_CENTRAL','DURATION_PERIPHERAL']

  data = pd.DataFrame()

  for scp in allTrials:
    scp['CURRENT_FIX_INDEX'] = scp['CURRENT_FIX_INDEX'].map(roi_dic)
    
    features = pd.DataFrame(columns = cols+roi_cols)

    for c in cols:
      features.loc[0,c] = scp.loc[0,c]

    for c in roi_cols: 
      features.loc[0,c] = scp[c].sum()

    for i in range(number_of_fixations):
      features.loc[0,f'Fixation_{i+1}'] = scp.loc[i,'CURRENT_FIX_INDEX']

    data = pd.concat([data,features])
  
  data = data.reset_index(drop=True)
  data = data.drop(columns='CURRENT_FIX_INDEX')

  return data


# Training the Classical model on the DataFrame of Scanpaths
def trainData(data,trainList,testList):
  
  auc = []
  for i in range(len(testList)):
    # corss validation
      testDf = data.merge(pd.DataFrame(testList[i],columns=['RECORDING_SESSION_LABEL']),on='RECORDING_SESSION_LABEL')
      trainDf = data.merge(pd.DataFrame(trainList[i],columns=['RECORDING_SESSION_LABEL']),on='RECORDING_SESSION_LABEL')

      testDf = testDf.drop(columns=['RECORDING_SESSION_LABEL'])
      trainDf = trainDf.drop(columns=['RECORDING_SESSION_LABEL'])

      # shuffle
      trainDf = trainDf.sample(frac=1)
      testDf = testDf.sample(frac=1)

      trainX = trainDf.drop(columns=['LABEL'])
      trainy = trainDf['LABEL'].astype(float)
      testX = testDf.drop(columns=['LABEL'])
      testy = testDf['LABEL'].astype(float)

      # oversample training data with copies of data for the positive label
      trainX = pd.concat([trainX,trainX[trainX.index.isin(trainy[trainy == 1].index)]])
      trainy = pd.concat([trainy,trainy[trainy == 1]])

      clf = LogisticRegression()
      clf.fit(trainX, trainy)
      preds = clf.predict(testX)
      auc.append(aucScore(testy,preds))
# output is the average auc
  return sum(auc)/len(auc)

#gerenating scanpath images
def scanpath_image_gen(group,number_of_fixations,path):

  plt.style.use('dark_background')
  rcParams['figure.figsize'] = 5, 5
  cmap = LinearSegmentedColormap.from_list('name', ['white', 'dimgray'])
  
  dfRaw = getData(path=path,labels=[f'{group} - New ROI Data','Control - New ROI Data'],phases=['Learning','Target','Distractor'])
  allTrials = getTrialData(dfRaw,number_of_fixations)
  df = pd.concat(allTrials)

  # scale duration sizes
  df['CURRENT_FIX_DURATION_SQUARED'] = (df['CURRENT_FIX_DURATION']**2)/400

  folderName = f'Scanpath Images - {group} - {number_of_fixations}'

  if not os.path.exists(path+f'/{folderName}'):
    os.mkdir(path+f'/{folderName}')

  for subject in df.RECORDING_SESSION_LABEL.unique():
    if not os.path.exists(path+f'/{folderName}'+f'/{subject}'):
      os.mkdir(path+f'/{folderName}'+f'/{subject}')
    subject_trials = df[df.RECORDING_SESSION_LABEL == subject]

    for trial in subject_trials.TRIAL_LABEL.unique():
      trialData = subject_trials[subject_trials.TRIAL_LABEL == trial].iloc[::-1]
      trialNum = trialData.TRIAL_LABEL.unique()[0]
      
      plt.clf()

      plt.xlim(345,675)
      plt.ylim(600,150)
      # face outline
      pts = np.array([[372,201], [649,201], [655,422],[607,530],[511,576],[416,530],[364,422]])
      p = Polygon(pts,ec='white',fc='none')
      ax = plt.gca()
      ax.add_patch(p)
      # fixations
      plt.scatter(trialData.CURRENT_FIX_X,trialData.CURRENT_FIX_Y,edgecolor='none',s=trialData.CURRENT_FIX_DURATION_SQUARED,c=trialData.index,cmap=cmap)
      plt.axis('off')
      plt.savefig(path+f'/{folderName}'+f'/{subject}'+f'/{trialNum}'+'.png')
      
      
# generating ROI sequence images
def sequence_image_gen(group,number_of_fixations,path):
    
  plt.style.use('dark_background')
  rcParams['figure.figsize'] = 5, 5

  dfRaw = getData(path=path,labels=[f'{group} - New ROI Data','Control - New ROI Data'],phases=['Learning','Target','Distractor'])
  allTrials = getTrialData(dfRaw,number_of_fixations)
  df = pd.concat(allTrials)
  
  rois= ['Forehead','LeftEyebrow','RightEyebrow', 'LeftEye', 'RightEye', 'LeftCheek', 'RightCheek', 'Nose', 'Mouth','Chin']
  cmap = LinearSegmentedColormap.from_list('name', ['black','white'])
  folderName = f'Temporal Images - {group} - {number_of_fixations}'

  if not os.path.exists(path+f'/{folderName}'):
    os.mkdir(path+f'/{folderName}')

  for subject in df.RECORDING_SESSION_LABEL.unique():
    if not os.path.exists(path+f'/{folderName}'+f'/{subject}'):
      os.mkdir(path+f'/{folderName}'+f'/{subject}')
    subject_trials = df[df.RECORDING_SESSION_LABEL == subject]
    subjectName = subject_trials.RECORDING_SESSION_LABEL.unique()[0]

    for trial in subject_trials.TRIAL_LABEL.unique():
      trialData = subject_trials[subject_trials.TRIAL_LABEL == trial]
      trialNum = trialData.TRIAL_LABEL.unique()[0]

      arr = np.array(trialData[rois])[0]
        
      for i in range(1,number_of_fixations):
        arr = np.vstack((arr,np.array(trialData[rois])[i]))

      plt.clf()
      sns.heatmap(arr,cbar=False,xticklabels=False,yticklabels=False,cmap=cmap)
      plt.margins(0)
      plt.axis('off')
      plt.savefig(path+f'/{folderName}'+f'/{subject}'+f'/{trialNum}'+'.png',bbox_inches = 'tight',pad_inches=0)
      
      
# Splitting the Data to Train (,Validation) and Test Set for the CNN
def getTrainTest(path,positiveClass='AP',valSet=False,getSubjectsFrom='Folder',imageFolder='none',df='none'):
  train = []
  val = []
  test = [] 
  negative = []
  positive = []

  if getSubjectsFrom=='Folder':
      allSubjects = sorted(os.listdir(f'{path}/{imageFolder}'))
  elif getSubjectsFrom == 'DataFrame':
      allSubjects = df.RECORDING_SESSION_LABEL.unique().tolist()
  
  
  for s in allSubjects:
    if s[-4:] == 'cont':
      negative.append(s)
    else:
      positive.append(s)
  
  r = len(positive)

  
  index= 0
  first = True

  if valSet == True:
    for i in range(r):
      if i == r-1:
        positive_val = [positive[i]]
        positive_test = [positive[0]]
      else:
        positive_val = [positive[i]]
        positive_test = [positive[i+1]]
      
      if positiveClass == 'DP':
        try:
          negative_val = [negative[index],negative[index+1]]
        except IndexError:
          negative_val = [negative[-1]]      

        try:
          negative_test = [negative[index+2],negative[index+3]]
        except IndexError:
          if first:
            negative_test = [negative[-1]]
            first = False
          else:
            negative_test = [negative[0],negative[1]]
        
        index += 2
      
      else:
          negative_val = [negative[index],negative[index+1]]
          negative_test = [negative[index+2],negative[index+3]]
          index += 2

      positive_train = list(set(positive).difference(set(positive_test+positive_val)))
      negative_train = list(set(negative).difference(set(negative_test+negative_val)))

        
      train.append(positive_train+negative_train)
      val.append(positive_val+negative_val)
      test.append(positive_test+negative_test)

    return train, val, test
    
  else:
    index = 0
    for i in range(r):
      positive_test = [positive[i]]
      negative_test = [negative[index],negative[index+1]]
      index += 2
        
      positive_train = list(set(positive).difference(set(positive_test)))
      negative_train = list(set(negative).difference(set(negative_test)))
        
      train.append(positive_train+negative_train)
      test.append(positive_test+negative_test)
    
    return train, test

# Converting Images Of Scanpaths to Arrays for Training the CNNs
class Image2Data:
    
  def __init__(self,imageType,listSubjects,path,imageFolder):
    self.imageType = imageType
    self.listSubjects = listSubjects
    self.imageFolder = imageFolder 
    X = []
    y = []
    names = []
    trials = []
    for subject in self.listSubjects:
      p = f"{path}/{self.imageFolder}/{subject}"
      subjectImages = os.listdir(p)
      random.shuffle(subjectImages)

      for img in subjectImages:
        arr = np.asarray(Image.open(p+'/'+img).convert(f'{self.imageType}'))
        X.append(arr)
        if subject[-4:] == 'cont':
          y.append(0)
        else: 
          y.append(1)
        names.append(subject)
        trialName = trials.append(img)

    self.X = np.array(X)
    self.y = np.array(y)
    self.names = names
    self.trials = trials
    
  def oversample(self):
    self.X = np.concatenate([self.X,self.X[np.where(self.y == 1)[0]]],axis=0)
    self.y = np.concatenate([self.y,self.y[np.where(self.y == 1)[0]]])

  def shuffle(self):
    shuffler = np.random.permutation(len(self.X))
    self.X = self.X[shuffler]
    self.y = self.y[shuffler]

# CNN Model Architecture 
def getModel(lr,input_shape,convlayer,convdim):
    
  classifier = Sequential()
  for i in range(convlayer):
    if i == 0:
      classifier.add(Convolution2D(convdim[i], 3, 3, input_shape = input_shape, activation = 'relu'))
      classifier.add(MaxPooling2D(pool_size = (2,2)))
    else:
      classifier.add(Convolution2D(convdim[i], 3, 3, activation = 'relu'))
      classifier.add(MaxPooling2D(pool_size = (2,2)))
  classifier.add(Flatten())
  classifier.add(Dense(64, activation = 'tanh'))
  classifier.add(Dropout(0.5))
  classifier.add(Dense(16, activation = 'tanh'))
  classifier.add(Dense(1, activation = 'sigmoid'))
  classifier.compile(optimizer = keras.optimizers.Adam(lr), loss = 'binary_crossentropy', metrics = ['accuracy'])
  return classifier

# Training the CNN Model
def trainModel(model,trainX,trainy,valX,valy,batch_size,patience):
  earlystopping = callbacks.EarlyStopping(monitor ="val_accuracy", mode ="max", patience = patience, restore_best_weights = True)
  model.fit(trainX, trainy, steps_per_epoch=len(trainX) // batch_size,batch_size = batch_size, epochs=30,validation_data=(valX,valy),callbacks =[earlystopping])
  return model
  

# Different Evaluation Metrics
def getMetrics(model,X,y,metric_type):
  if metric_type == 'acc':
    return model.evaluate(np.array(X),y)[1]

  elif metric_type == 'auc':
    preds = model.predict(np.array(X))
    return aucScore(y,preds)

  elif metric_type == 'recall':
    target_names = ['negative', 'positive']
    preds = model.predict(np.array(X))
    report = classification_report(y, preds, target_names=target_names, output_dict=True)
    return report['negative']['recall'], report['positive']['recall']

# Plotting Loss and Accuracy
def getPlots(hist):
    
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
def getValResults(imageFolder,path, modelName,positiveClass,patience):
      
  results = {'train_set':[],'val_set':[],'test_set':[],'train_acc':[],'train_auc':[],'val_acc':[],'val_auc':[]} 
  train, val, test = getTrainTest(path,positiveClass=positiveClass,valSet=True,imageFolder=imageFolder)

  for i in range(len(test)):

      results['train_set'].append(train[i])
      results['val_set'].append(val[i])
      results['test_set'].append(test[i])

      trainData = Image2Data('RGB', train[i], path, imageFolder)
      trainData.oversample()
      trainData.shuffle()
      trainX, trainy  = trainData.X, trainData.y 

      valData = Image2Data('RGB', val[i], path, imageFolder)
      valX, valy  = valData.X, valData.y 

      model = getModel(lr=0.0001,input_shape=trainX.shape[1:],convlayer=3,convdim=[32,64,128]) 
      model = trainModel(model,trainX, trainy, valX, valy, batch_size=32,patience=patience)
      

      results['train_acc'].append(getMetrics(model,trainX, trainy,'acc'))
      results['train_auc'].append(getMetrics(model,trainX, trainy,'auc'))
      print('train acc:',getMetrics(model,trainX, trainy,'acc'),' train auc:',getMetrics(model,trainX, trainy,'auc'))

      results['val_acc'].append(getMetrics(model,valX, valy,'acc'))
      results['val_auc'].append(getMetrics(model,valX, valy,'auc'))
      print('val acc:',getMetrics(model,valX, valy,'acc'),' val auc:',getMetrics(model,valX, valy,'auc'))

      model.save(f'{path}/{modelName}_{i}')

  return pd.DataFrame(results)

# Getting the CNN Model Results using the Test Set
def getTestResults(imageFolder,path,modelName,positiveClass):
      
  results = {'test_acc':[],'test_auc':[],'negative_class_ratio':[],'positive_class_ratio':[]} 
  _, _, test = getTrainTest(path,positiveClass=positiveClass,valSet=True,imageFolder=imageFolder)

  for i in range(len(test)):

    testData = Image2Data('RGB', test[i], path, imageFolder)
    testX, testy  = testData.X, testData.y

    results['negative_class_ratio'].append(len(np.where(testy == 0)[0])/len(testy))
    results['positive_class_ratio'].append(len(np.where(testy == 1)[0])/len(testy))

    model = keras.models.load_model(f'{path}/{modelName}_{i}')
    preds = model.predict(testX)  
    
    results['test_acc'].append(getMetrics(model,testX, testy,'acc'))
    results['test_auc'].append(getMetrics(model,testX, testy,'auc'))

  return pd.DataFrame(results)
  
# visualizing the group results
def groupResults(path,imageFolder,modelName,positiveClass,temporalModel,runNumber=None):

  _ ,_ , test = getTrainTest(path,positiveClass, valSet=True,imageFolder=imageFolder)
  positive = []
  negative = [] 

  factor =  1/4 
  fixations = 16 if positiveClass == 'DP' else 4

  if runNumber != None:
    for i in range(len(test)):
      testData = Image2Data('RGB', test[i], path, imageFolder)
      testX, testy  = testData.X, testData.y
      model = keras.models.load_model(f'{path}/{positiveClass} Models/model_{modelName}_run_{runNumber}_{i}')
      preds = (model.predict(testX)).reshape(testy.shape[0],1)

      for p in range(len(preds)):
        if preds[p] >= np.percentile(preds, 90):
          positive.append(testX[p])

        elif preds[p] <= np.percentile(preds, 10):
          negative.append(testX[p])

  else:
    for i in range(len(test)):
      testData = Image2Data('RGB', test[i], path, imageFolder)
      testX, testy  = testData.X, testData.y
      
      sumPreds = np.zeros((testy.shape[0],1))
      for j in range(10):
        model = keras.models.load_model(f'{path}/{positiveClass} Models/model_{modelName}_run_{j}_{i}')
        preds = (model.predict(testX)).reshape(testy.shape[0],1)
        sumPreds += preds

      meanPreds = sumPreds/10

      for p in range(len(meanPreds)):
        if meanPreds[p] >= np.percentile(meanPreds, 90):
          positive.append(testX[p])
        elif meanPreds[p] <= np.percentile(meanPreds, 10):
          negative.append(testX[p])    

  negativeImages = np.zeros((negative[0].shape[0],negative[0].shape[1],3))
  for n in negative:
    negativeImages += n
  negativeImages = negativeImages ** factor

  positiveImages = np.zeros((positive[0].shape[0],positive[0].shape[1],3))
  for p in positive:
    positiveImages += p
  positiveImages = positiveImages ** factor
  

  xTicksRange = np.arange(negative[0].shape[1]/20,negative[0].shape[1],negative[0].shape[1]/10)
  xTickLabels = ['Forehead','LeftEyebrow','RightEyebrow', 'LeftEye', 'RightEye', 'LeftCheek', 'RightCheek', 'Nose', 'Mouth','Chin']
  yTicksRange = np.arange(negative[0].shape[0]/(fixations*2),negative[0].shape[0],negative[0].shape[0]/fixations)
  yTickLabels = [f'Fixation {i}' for i in range(1,fixations+1)]


  fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,15))
  if temporalModel == True:
    images = [positiveImages,negativeImages]
    for i in range(2):
      axes[i].imshow(array_to_img(images[i]))
      axes[i].set_xticks(ticks = xTicksRange)
      axes[i].set_yticks(ticks = yTicksRange)
      axes[i].set_xticklabels(xTickLabels,rotation=35,fontsize=8)
      axes[i].set_yticklabels(yTickLabels,fontsize=8)      
  else:
    cmap = 'inferno'
    axes[0].matshow(positiveImages[:,:,0],cmap=cmap)
    axes[1].matshow(negativeImages[:,:,0],cmap=cmap)
    for i in range(2):
      axes[i].axis('off') 
  
  return positiveImages, negativeImages
 
# visualizing an individual subjects results
def subjectResults(path,testSet,imageFolder,positiveClass,temporalModel=False):
  _ ,_ , test = getTrainTest(path,positiveClass, valSet=True,imageFolder=imageFolder)
  
  testData = Image2Data('RGB', test[testSet], path, imageFolder)
  testX, testy  = testData.X, testData.y
  
  X = []
  for i in range(len(testy)):
    if testy[i] == 1:
      X.append(testX[i])
  factor = 1/4

  allImages = np.zeros((X[0].shape[0],X[0].shape[1],3))
  for n in X:
    allImages += n

  fixations = 16 if positiveClass == 'DP' else 4
  xTicksRange = np.arange(X[0].shape[1]/20,X[0].shape[1],X[0].shape[1]/10)
  xTickLabels = ['Forehead','LeftEyebrow','RightEyebrow', 'LeftEye', 'RightEye', 'LeftCheek', 'RightCheek', 'Nose', 'Mouth','Chin']
  yTicksRange = np.arange(X[0].shape[0]/(fixations*2),X[0].shape[0],X[0].shape[0]/fixations)
  yTickLabels = [f'Fixation {i}' for i in range(1,fixations+1)]


  plt.figure(figsize=(7,7))
  if temporalModel:
    plt.imshow(array_to_img(allImages))
    plt.xticks(ticks = xTicksRange,labels=xTickLabels,rotation=35,fontsize=8)
    plt.yticks(ticks = yTicksRange,labels=yTickLabels,fontsize=8)
  
  else:
    plt.matshow(allImages[:,:,0]**factor,fignum=1,cmap="inferno")
    plt.axis('off')
  
  return allImages
 
# creating heatmaps of the 10 ROIs
def faceHeatmap(path,image,positiveClass,section):
  if positiveClass == 'AP':
    skip = 68
    n_samples = list(image[:,:,0][0+(section)*skip,::28].astype(int)**2)+[1,1,1,1]
  
  else:
    skip = 18
    sectionImages = np.zeros((10,))
    start = section*4
    end = (section*4) + 4
    for j in range(start,end):
      sectionImages += image[:,:,0][0+(j)*skip,::28]
    aveImage = sectionImages/4
    n_samples = list(aveImage.astype(int)**2)+[1,1,1,1]
    
    
  plt.figure()
  X, _ = make_blobs(n_samples=n_samples, centers=[[510,243],[424,275],[611,275], #forhead, lefteyebrow, righteyebrow
                                                                                [442,330],[580,330], #lefteye, righteye
                                                                                [398,400],[641,400], #leftcheek, rightcheek
                                                                                [512,392],[512,490],[512,570],# nose, mouth, chin
                                                                                [350,350],[650,350],[500,150],[500,600]], # to control image dimensions
                                                                                cluster_std=0.5)             
  plt.xlim(345,675)
  plt.ylim(600,150)
  sns.kdeplot(x=X[:,0], y=X[:,1],fill=True, thresh=0, levels=100, cmap="inferno",zorder=1)

  plt.axis('off')
  plt.imshow(mpimg.imread(path+'/1a.bmp'),zorder=1,alpha=0.6)

# Calculate root mean squared contrast
def rmsContrast(arr):
  meanIntensity = np.mean(arr)
  S = (arr-meanIntensity)**2
  MS = np.mean(S)
  RMS = MS**1/2
  return RMS
  
# training random hyrbid models 
def randomHybridAUC(path,imageFolder,positiveClass,patience):
  train, val, test = getTrainTest(path,positiveClass=positiveClass,valSet=True,imageFolder=imageFolder)
  auc = []
  
  for i in range(len(test)): 
      # Getting Random Scanpath Predictions for ith Cross Validation step
      trainData = Image2Data('RGB', train[i], path, f'Scanpath Images - {positiveClass}')
      trainData.oversample()
      trainData.shuffle()
      trainX, trainy  = trainData.X, trainData.y 
      np.random.shuffle(trainy)

      valData = Image2Data('RGB', val[i], path, f'Scanpath Images - {positiveClass}')
      valX, valy  = valData.X, valData.y 
      
      testData = Image2Data('RGB', test[i], path,f'Scanpath Images - {positiveClass}')
      testX, testy  = testData.X, testData.y 
      
      scanModel = getModel(lr=0.0001,input_shape=trainX.shape[1:],convlayer=3,convdim=[32,64,128]) 
      scanModel = trainModel(scanModel,trainX, trainy, valX, valy, batch_size=32,patience=patience)
      scanPreds = scanModel.predict(testX)     

      # Getting Random Temporal Predictions for ith Cross Validation step
      trainData = Image2Data('RGB', train[i], path, f'Temporal Images - {positiveClass}')
      trainData.oversample()
      trainData.shuffle()
      trainX, trainy  = trainData.X, trainData.y 
      np.random.shuffle(trainy)

      valData = Image2Data('RGB', val[i], path, f'Temporal Images - {positiveClass}')
      valX, valy  = valData.X, valData.y 
      
      testData = Image2Data('RGB', test[i], path, f'Temporal Images - {positiveClass}')
      testX, testy  = testData.X, testData.y 
    

      tempModel = getModel(lr=0.0001,input_shape=trainX.shape[1:],convlayer=3,convdim=[32,64,128]) 
      tempModel = trainModel(tempModel,trainX, trainy, valX, valy, batch_size=32,patience=patience)
      tempPreds = tempModel.predict(testX)
      

      aucList = []
      weights = np.linspace(0,1,1000)
      for w in weights:
        iw = 1-w
        aucList.append(aucScore(testy,(w * scanPreds+ iw * tempPreds))) 

      auc.append(max(aucList))
      
  return (sum(auc)/len(auc))