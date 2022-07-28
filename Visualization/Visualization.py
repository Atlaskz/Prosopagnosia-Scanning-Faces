
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






# # visualizing the group results
# def GroupResults(path,image_folder,model_name,positive_class,temporal_model,num_fixations=num_fixations,run_number=None):

#   _ ,_ , test = TrainTestSplit(path,positive_class, validation=True,image_folder=image_folder)
#   positive = []
#   negative = [] 

#   factor =  1/4 

#   if run_number != None:
#     for i in range(len(test)):
#       test_data = Image2Array('RGB', test[i], path, image_folder)
#       test_X, test_y  = test_data.X, test_data.y
#       model = keras.models.load_model(f'{path}/{positive_class} Models/model_{model_name}_run_{run_number}_{i}')
#       preds = (model.predict(test_X)).reshape(testy.shape[0],1)

#       for p in range(len(preds)):
#         if preds[p] >= np.percentile(preds, 90):
#           positive.append(test_X[p])

#         elif preds[p] <= np.percentile(preds, 10):
#           negative.append(test_X[p])

#   else:
#     for i in range(len(test)):
#       test_data = Image2Array('RGB', test[i], path, image_folder)
#       test_X, test_y  = test_data.X, test_data.y
      
#       sum_preds = np.zeros((testy.shape[0],1))
#       for j in range(10):
#         model = keras.models.load_model(f'{path}/{positive_class} Models/model_{model_name}_run_{j}_{i}')
#         preds = (model.predict(testX)).reshape(testy.shape[0],1)
#         sum_preds += preds

#       mean_preds = sum_preds/10

#       for p in range(len(mean_preds)):
#         if mean_preds[p] >= np.percentile(mean_preds, 90):
#           positive.append(test_X[p])
#         elif mean_preds[p] <= np.percentile(mean_preds, 10):
#           negative.append(test_X[p])    

#   negative_images = np.zeros((negative[0].shape[0],negative[0].shape[1],3))
#   for n in negative:
#     negative_images += n
#   negative_images = negative_images ** factor

#   positive_images = np.zeros((positive[0].shape[0],positive[0].shape[1],3))
#   for p in positive:
#     positive_images += p
#   positive_images = positive_images ** factor
  

#   xTicksRange = np.arange(negative[0].shape[1]/20,negative[0].shape[1],negative[0].shape[1]/10)
#   xTickLabels = ['Forehead','LeftEyebrow','RightEyebrow', 'LeftEye', 'RightEye', 'LeftCheek', 'RightCheek', 'Nose', 'Mouth','Chin']
#   yTicksRange = np.arange(negative[0].shape[0]/(fixations*2),negative[0].shape[0],negative[0].shape[0]/num_fixations)
#   yTickLabels = [f'Fixation {i}' for i in range(1,fixations+1)]


#   fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,15))
#   if temporal_model == True:
#     images = [positive_images,negative_images]
#     for i in range(2):
#       axes[i].imshow(array_to_img(images[i]))
#       axes[i].set_xticks(ticks = xTicksRange)
#       axes[i].set_yticks(ticks = yTicksRange)
#       axes[i].set_xticklabels(xTickLabels,rotation=35,fontsize=8)
#       axes[i].set_yticklabels(yTickLabels,fontsize=8)      
#   else:
#     cmap = 'inferno'
#     axes[0].matshow(positive_images[:,:,0],cmap=cmap)
#     axes[1].matshow(negative_images[:,:,0],cmap=cmap)
#     for i in range(2):
#       axes[i].axis('off') 
  
#   return positive_images, negative_images
 
 
 
 
 
 
 
 
 
 
# # visualizing an individual subjects results
# def SubjectResults(path,testSet,imageFolder,positiveClass,temporalModel=False):
#   _ ,_ , test = TrainTestSplit(path,positive_class, validation=True,image_folder=image_folder)
  
#   testData = Image2Data('RGB', test[testSet], path, imageFolder)
#   testX, testy  = testData.X, testData.y
  
#   X = []
#   for i in range(len(testy)):
#     if testy[i] == 1:
#       X.append(testX[i])
#   factor = 1/4

#   allImages = np.zeros((X[0].shape[0],X[0].shape[1],3))
#   for n in X:
#     allImages += n

#   fixations = 16 if positiveClass == 'DP' else 4
#   xTicksRange = np.arange(X[0].shape[1]/20,X[0].shape[1],X[0].shape[1]/10)
#   xTickLabels = ['Forehead','LeftEyebrow','RightEyebrow', 'LeftEye', 'RightEye', 'LeftCheek', 'RightCheek', 'Nose', 'Mouth','Chin']
#   yTicksRange = np.arange(X[0].shape[0]/(fixations*2),X[0].shape[0],X[0].shape[0]/fixations)
#   yTickLabels = [f'Fixation {i}' for i in range(1,fixations+1)]


#   plt.figure(figsize=(7,7))
#   if temporalModel:
#     plt.imshow(array_to_img(allImages))
#     plt.xticks(ticks = xTicksRange,labels=xTickLabels,rotation=35,fontsize=8)
#     plt.yticks(ticks = yTicksRange,labels=yTickLabels,fontsize=8)
  
#   else:
#     plt.matshow(allImages[:,:,0]**factor,fignum=1,cmap="inferno")
#     plt.axis('off')
  
#   return allImages

 
 
 
 
# # creating heatmaps of the 10 ROIs
# def FaceHeatmap(path,image,positiveClass,section):
#   if positiveClass == 'AP':
#     skip = 68
#     n_samples = list(image[:,:,0][0+(section)*skip,::28].astype(int)**2)+[1,1,1,1]
  
#   else:
#     skip = 18
#     sectionImages = np.zeros((10,))
#     start = section*4
#     end = (section*4) + 4
#     for j in range(start,end):
#       sectionImages += image[:,:,0][0+(j)*skip,::28]
#     aveImage = sectionImages/4
#     n_samples = list(aveImage.astype(int)**2)+[1,1,1,1]
    
    
#   plt.figure()
#   X, _ = make_blobs(n_samples=n_samples, centers=[[510,243],[424,275],[611,275], #forhead, lefteyebrow, righteyebrow
#                                                                                 [442,330],[580,330], #lefteye, righteye
#                                                                                 [398,400],[641,400], #leftcheek, rightcheek
#                                                                                 [512,392],[512,490],[512,570],# nose, mouth, chin
#                                                                                 [350,350],[650,350],[500,150],[500,600]], # to control image dimensions
#                                                                                 cluster_std=0.5)             
#   plt.xlim(345,675)
#   plt.ylim(600,150)
#   sns.kdeplot(x=X[:,0], y=X[:,1],fill=True, thresh=0, levels=100, cmap="inferno",zorder=1)

#   plt.axis('off')
#   plt.imshow(mpimg.imread(path+'/1a.bmp'),zorder=1,alpha=0.6)

# # Calculate root mean squared contrast
# def rmsContrast(arr):
#   meanIntensity = np.mean(arr)
#   S = (arr-meanIntensity)**2
#   MS = np.mean(S)
#   RMS = MS**1/2
#   return RMS
  


def PlotFace(image_paths,fig_path):
    
    factor = 1/4
    
    image_list = []
    for path in image_paths:
        try:
            image_list.append(np.asarray(Image.open(path).convert('RGB')))
        except FileNotFoundError:
            pass
    print(image_list)
    
    images = np.zeros((image_list[0].shape[0],image_list[0].shape[1],3))
    for i in image_list:
        images += i
    images = images ** factor
    
    fig, axis = plt.subplots(nrows=1, ncols=1,figsize=(15,15))
    cmap = 'inferno'
    axis.matshow(images[:,:,0],cmap=cmap)
    fig.savefig(fig_path)