from analysis.analysis_tools.loading import *
from analysis.analysis_tools.training import *
from analysis.analysis_tools.processing import *
from analysis.image_classification.classification_tools import *
import keras
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge


def HybridDf(path,labels,num_fixations):

    '''
    Creates the dataframe passed to the hybrid model for regression
    
    Args:
        path : path to root directory
        labels: dictionary of the form {positive_class : *name of positive class*, negative_class : *name of negative class*} 
        num_fixations: number of fixations used to generaate the images
    
    Output
        dataframe with columns as the subject names, true label and the predected label (as a probability) from the temporal and the scanpath models 
    '''
    
    preds_dict = {'subject':[],'ytrue':[],'scanpath_model_preds':[],'temporal_model_preds':[]}
    model_list = ['scanpath','temporal']
    
    
    data_path = os.path.join(path,'data') 
    p_subjects = EyeTrackingData(data_path = data_path,labels={'positive_class': labels['positive_class']}).Names()
    n_subjects = EyeTrackingData(data_path = data_path,labels={'negative_class': labels['negative_class']}).Names()
    
    
    _, _, test = CreateSplits(positive_class=labels['positive_class'], 
                            positive_subjects=p_subjects, 
                            negative_subjects=n_subjects,
                            validation=True, 
                            return_list = False)
    
    for test_subjects in test:
        fold = 0
    
        for model_name in model_list:
          
            image_folder = f"{model_name} - {labels['positive_class']} - {num_fixations}"
            test_data = Image2Array(subjects=test_subjects, 
                                  path=path, 
                                  labels=labels, 
                                  image_folder= image_folder,
                                  num_fixations=num_fixations)
            
            
            
            test_X, test_y  = test_data.X, test_data.y
            all_runs = []
            for run_number in range(1,11):
                model = keras.models.load_model(os.path.join(path, 'models', labels['positive_class'], f'{model_name}_run_{run_number}_subject_{fold}'))
                preds = (model.predict(test_X)).reshape(test_y.shape[0],1)  
                all_runs.append(preds)
            mean_preds = sum(all_runs)/len(all_runs)

            preds_dict[f'{model_name}_model_preds'].extend(mean_preds)
        
        preds_dict['ytrue'].extend(test_y)
        preds_dict['subject'].extend(test_data.names)
        fold += 1

    
    preds_df =pd.DataFrame.from_dict(preds_dict)
    
    return preds_df, test



def HybridModel(path,labels,num_fixations,model_weights='manual_random'):
  
    '''
    Finds the best combination of weights between the scanpath and temporal model predictions to maximize the area under the curve  
    
    Args:
        path: path to root directory
        labels: dictionary of the form {positive_class : *name of positive class*, negative_class : *name of negative class*} 
        num_fixations: number of fxations used to generate the images
        model_weights: how weights are calculated. Can choose between manual_random and lin_reg.
    Output:
        dataframe containing the weight used for each model (scanpath and temporal) in each classification fold and the resulting auc for that fold '''
    
    preds_df, test = HybridDf(path,labels,num_fixations)
    
    test_list = []
    for test_dict in test:
        test_list.append(test_dict['positive_class']+test_dict['negative_class'])
        
    hybrid= {'test':[],'temporal_model_weight':[],'scanpath_model_weight':[],'auc':[],
              'positive_class_ratio':[],'negative_class_ratio':[]}

    print(len(test_list))
    for i in range(len(test_list)):
      
        tmp_df = preds_df[preds_df.subject.isin(test_list[i])]
        X = tmp_df.drop(columns=['subject','ytrue'])
        y = tmp_df.ytrue

        if model_weights == 'lin_reg':
        
            reg = LinearRegression().fit(X,y)
            preds = reg.predict(X)
            auc_score = aucScore(y,preds)
            
            hybrid['temporal_model_weight'].append(reg.coef_[1])
            hybrid['scanpath_model_weight'].append(reg.coef_[0])
            hybrid['auc'].append(auc_score)
            hybrid['test'].append(test_list[i])
            hybrid['positive_class_ratio'].append(len(tmp_df[tmp_df.ytrue==1])/len(tmp_df))
            hybrid['negative_class_ratio'].append(len(tmp_df[tmp_df.ytrue==0])/len(tmp_df))
    
        elif model_weights == 'manual_random':
              
            tmp_dict = {'scanpath':[],'temporal':[],'auc':[]}
            weights = np.linspace(0,1,1000)
            
            for w in weights:
                iw = 1-w
                tmp_dict['scanpath'].append(w)
                tmp_dict['temporal'].append(iw)
                tmp_dict['auc'].append(aucScore(y,(w * tmp_df['scanpath_model_preds']+iw * tmp_df['temporal_model_preds']))) 
            
            max_auc = pd.DataFrame.from_dict(tmp_dict).sort_values(by='auc',ascending=False).iloc[0,:]
            preds = max_auc['scanpath'] * tmp_df['scanpath_model_preds'] + max_auc['temporal'] * tmp_df['temporal_model_preds']
            
            hybrid['temporal_model_weight'].append(max_auc['temporal'])
            hybrid['scanpath_model_weight'].append(max_auc['scanpath'])
            hybrid['auc'].append(max_auc['auc'])
            hybrid['test'].append(test_list[i])
            hybrid['positive_class_ratio'].append(len(tmp_df[tmp_df.ytrue==1])/len(tmp_df))
            hybrid['negative_class_ratio'].append(len(tmp_df[tmp_df.ytrue==0])/len(tmp_df))
    
    return pd.DataFrame.from_dict(hybrid)

    
    
    
    
# training random hyrbid models 
def RandomHybridModel(path,labels,num_fixations):
    
    '''
    Create a hybrid model with random weights. Used for comparison with the original hybrid models results
    '''
    data_path = os.path.join(path,'data') 
    p_subjects = EyeTrackingData(data_path = data_path,labels={'positive_class': labels['positive_class']}).Names()
    n_subjects = EyeTrackingData(data_path = data_path,labels={'negative_class': labels['negative_class']}).Names()
    
    train, val, test = CreateSplits(labels['positive_class'], p_subjects, n_subjects,validation=True,return_list=False)
    auc = []
    
    for i in range(len(test)): 
        # Getting Random Scanpath Predictions for ith Cross Validation step
        image_folder = f"scanpath - {labels['positive_class']} - {num_fixations}"
        trainData = Image2Array(subjects=train[i], 
                                  path=path, 
                                  labels=labels, 
                                  image_folder= image_folder,
                                  num_fixations=num_fixations)
        trainData.upsample()
        trainData.shuffle()
        train_X, train_y  = trainData.X, trainData.y 
        np.random.shuffle(train_y)
        
        valData = Image2Array(subjects=val[i], 
                                  path=path, 
                                  labels=labels, 
                                  image_folder= image_folder,
                                  num_fixations=num_fixations)
        val_X, val_y  = valData.X, valData.y 
        
        testData = Image2Array(subjects=test[i], 
                                  path=path, 
                                  labels=labels, 
                                  image_folder= image_folder,
                                  num_fixations=num_fixations)
        test_X, test_y  = testData.X, testData.y 
        
        scanModel = Model(lr=0.0001,input_shape=train_X.shape[1:],convlayer=3,convdim=[32,64,128]) 
        scanModel = FitModel(scanModel,train_X, train_y, val_X, val_y, batch_size=32,patience=10)
        scanPreds = scanModel.predict(test_X)     
        
        # Getting Random Temporal Predictions for ith Cross Validation step
        image_folder = f"temporal - {labels['positive_class']} - {num_fixations}"
        trainData = Image2Array(subjects=train[i], 
                                  path=path, 
                                  labels=labels, 
                                  image_folder= image_folder,
                                  num_fixations=num_fixations)
        trainData.upsample()
        trainData.shuffle()
        train_X, train_y  = trainData.X, trainData.y 
        np.random.shuffle(train_y)
        
        valData = Image2Array(subjects=val[i], 
                                  path=path, 
                                  labels=labels, 
                                  image_folder= image_folder,
                                  num_fixations=num_fixations)
        val_X, val_y  = valData.X, valData.y 
        
        testData = Image2Array(subjects=test[i], 
                                  path=path, 
                                  labels=labels, 
                                  image_folder= image_folder,
                                  num_fixations=num_fixations)
        test_X, test_y  = testData.X, testData.y 
        
        
        tempModel = Model(lr=0.0001,input_shape=train_X.shape[1:],convlayer=3,convdim=[32,64,128]) 
        tempModel = FitModel(tempModel,train_X, train_y, val_X, val_y, batch_size=32,patience=patience)
        tempPreds = tempModel.predict(testX)
        
        
        aucList = []
        weights = np.linspace(0,1,1000)
        for w in weights:
            iw = 1-w
            aucList.append(aucScore(test_y,(w * scanPreds+ iw * tempPreds))) 
        
        auc.append(max(aucList))

    return (sum(auc)/len(auc))