# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:07:34 2018

@author: fuyang
"""

import pandas as pd
import time
import numpy as np
import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
import gc
from contextlib import contextmanager
import _pickle as cPickle
from sklearn.linear_model import LogisticRegression
import os
#import pathlib

'''
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
'''




#====================================================================body    
dataDir = '~/527/'    #data also in my udrive



#===============bayes opt for hyper param
"""
def status_print(optim_result):
    #Status callback durring bayesian hyperparameter search
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    #best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")


HYPER_TUNE_SIZE = 20000000

#with timer('load training data'):
#        train_df = pd.read_csv(dataDir+'train_new.csv',nrows=HYPER_TUNE_SIZE)
val_df = pd.read_csv(dataDir+'train_new.csv',nrows=HYPER_TUNE_SIZE)

target = 'is_attributed'
y = val_df[target].values
val_df.drop(['click_time','category','epochtime',target],axis=1,inplace=True)
gc.collect()

categorical_features = [i for i in range(val_df.shape[1]) if val_df.columns[i] in ['ip','app','os','channel','device']]
predictors = list(val_df.columns)



ITERATIONS = 20 
bayes_cv_tuner = BayesSearchCV(
    estimator = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_jobs=1,
        verbose=0,
        feature_name=predictors,
        categorical_feature=categorical_features
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'num_leaves': (3, 100),
        'max_depth': (2, 50),
        'min_child_samples': (0, 50),
        'max_bin': (100, 1000),
        'subsample': (0.5, 1.0, 'uniform'),
        'subsample_freq': (0, 10),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'min_child_weight': (0, 10),
        'subsample_for_bin': (100000, 1000000),
        'reg_lambda': (1e-9, 10, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'scale_pos_weight': (50, 500, 'log-uniform'),
        'n_estimators': (50, 100),
    },    
    scoring = 'roc_auc',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=6667
    ),
    n_jobs = 4,
    n_iter = ITERATIONS,   
    verbose = 0,
    refit = True,
    random_state = 6666
)
    
gc.collect()

#with timer('bayes opt on partial data'):
#    result = bayes_cv_tuner.fit(train_df[predictors].values, y, callback=status_print)

result = bayes_cv_tuner.fit(val_df[predictors].values, y, callback=status_print)

'''
Best params: {
'colsample_bytree': 0.5401, 
'learning_rate': 0.1247, 
'max_bin': 480,
'num_leaves': 56,
'max_depth': 50, 
'min_child_samples': 9, 
'min_child_weight': 7, 
'n_estimators': 92, 
'reg_alpha': 1.7843e-08, 
'reg_lambda': 6.4628e-08, 
'scale_pos_weight': 93.8484, 
'subsample': 0.9636, 
'subsample_for_bin': 428359, 
'subsample_freq': 9}
'''

"""
#====================================train chunk by chunk and average?
filename = dataDir+'models/model.p'
os.makedirs(os.path.dirname(filename), exist_ok=True)

lgb_params = {'boosting_type': 'gbdt',
              'objective': 'binary',
              'colsample_bytree': 0.5401, # Subsample ratio of columns when constructing each tree
              'learning_rate': 0.1247, 
              'max_bin': 480, # Number of bucketed bin for feature values
              'num_leaves': 56,
              'max_depth': 50, 
              'min_child_samples': 9, # Minimum number of data need in a child(min_data_in_leaf)
              'min_child_weight': 7, # Minimum sum of instance weight(hessian) needed in a child(leaf)
              'n_estimators': 92, 
              'reg_alpha': 1.7843e-08, # L1 regularization term on weights
              'reg_lambda': 6.4628e-08, # L2 regularization term on weights
              'scale_pos_weight': 93.8484, 
              'subsample': 0.9636, # Subsample ratio of the training instance
              'subsample_for_bin': 428359, # Number of samples for constructing bin
              'subsample_freq': 9, # frequence of subsample, <=0 means no enable
              'verbose': 0,
              'metric': 'auc',
              'n_jobs': 4
              }


chunksize = 20000000 
count = 0
for chunk in pd.read_csv(dataDir+'train_new.csv', chunksize=chunksize):
    target = 'is_attributed'
    y = chunk[target].values
    chunk.drop(['click_time','category','epochtime','is_attributed'],axis=1,inplace=True)
    gc.collect()
        
    categorical_features = [i for i in range(chunk.shape[1]) if chunk.columns[i] in ['ip','app','os','channel','device']]
    predictors = list(chunk.columns)
    
    if count == 0:
        X_valid = chunk.iloc[:int(chunksize)][predictors].values
        y_valid = y[:int(chunksize)]
        
        del chunk,y
        gc.collect()
        print('Finish loading validation chunk!')
        count += 1
        continue
    
#    with timer('Training chunk '+str(count)):     
    if True:
        print('Start training chunk',count,'!')
        
        #iid bagging model
        model_iid = lgb.LGBMClassifier(boosting_type = lgb_params['boosting_type'],
                                       objective = lgb_params['objective'],
                                       colsample_bytree = lgb_params['colsample_bytree'],
                                       learning_rate = lgb_params['learning_rate'],
                                       max_bin = lgb_params['max_bin'],
                                       max_depth = lgb_params['max_depth'],
                                       min_child_samples = lgb_params['min_child_samples'],
                                       min_child_weight = lgb_params['min_child_weight'],
                                       n_estimators = lgb_params['n_estimators'],
                                       reg_alpha = lgb_params['reg_alpha'],
                                       reg_lambda = lgb_params['reg_lambda'],
                                       scale_pos_weight = lgb_params['scale_pos_weight'],
                                       subsample = lgb_params['subsample'],
                                       subsample_for_bin = lgb_params['subsample_for_bin'],
                                       subsample_freq = lgb_params['subsample_freq'],
                                       verbose = lgb_params['verbose'],
                                       n_jobs = lgb_params['n_jobs']
                                       )
        
        model_iid.fit(X = chunk.iloc[:][predictors].values, 
                      y = y,
                      feature_name=predictors,
                      categorical_feature=categorical_features,
                      eval_metric='auc',
                      eval_set=(X_valid,y_valid),
                      early_stopping_rounds=20                      
                      )
        
        #filename = dataDir+'models/model'+str(count)+'.p'
        #abspath = pathlib.Path(filename).absolute()
        
        with open(dataDir+'models/model'+str(count)+'.p', 'wb') as f:
            cPickle.dump(model_iid, f)
            
        count += 1


del chunk,y,X_valid,y_valid
gc.collect()

"""        
#==========================stacking
with timer('load validation data'):
        val_df = pd.read_csv(dataDir+'train_new.csv',nrows=chunksize)

target = 'is_attributed'
y_val = val_df[target].values
val_df.drop(['click_time','category','epochtime',target],axis=1,inplace=True)
gc.collect()

predictions = np.zeros([test_df.shape[0],count-1])
for i in range(1,count):
    with open('D:/uw/stat527/proj/models_iid/model'+str(i)+'.p', 'rb') as f:
        model_iid = cPickle.load(f)
    predictions[:,i-1] = model_iid.predict_proba(test_df.iloc[:][predictors].values)[:,1]

lr = LogisticRegression(C=1000000.0, random_state=0)
lr.fit(predictions, y_val)



#=====================================================test
with timer('load test data'):
    test_df = pd.read_csv(dataDir+'test_new.csv')

res = pd.DataFrame()
res['click_id'] = test_df['click_id'].values
test_df.drop(['click_time','category','epochtime','click_id'],axis=1,inplace=True)
gc.collect()

predictions = np.zeros([test_df.shape[0],count-1])
for i in range(1,count):
    with open('D:/uw/stat527/proj/models_iid/model'+str(i)+'.p', 'rb') as f:
        model_iid = cPickle.load(f)
    predictions[:,i-1] = model_iid.predict_proba(test_df.iloc[:][predictors].values)[:,1]

res['is_attributed']  = predictions.dot(lr.coef_.transpose())
res.to_csv(dataDir+'my_result_iid.csv', float_format='%.8f', index=False)

'''
res['is_attributed']  = predictions1
res.to_csv(dataDir+'my_result.csv', float_format='%.8f', index=False)
'''



#=====================feature importance
t = np.zeros([18])
for i in range(1,17):
    with open('D:/uw/stat527/proj/models_iid/model1.p', 'rb') as f:
        model_iid = cPickle.load(f)
    t += model_iid.feature_importances_




fi = pd.DataFrame(t.reshape([1,-1]))
fi.columns = model_iid.booster_.feature_name()
fi.to_csv('D:/uw/stat527/proj/feat_imp.txt')
"""
