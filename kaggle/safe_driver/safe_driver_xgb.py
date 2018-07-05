# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:17:15 2017

@author: dell
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import datetime

os.chdir('C:\\Users\\dell\\Desktop\\Safe driver')
train = pd.read_csv("train.csv")
cat_cols = [col for col in train.columns if '_cat' in col]

#one-hot
train = pd.get_dummies(train, columns=cat_cols)
predictors = [x for x in train.columns if x not in ['target', 'id']]


def xgbfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, dtest=None):
    
    starttime = datetime.datetime.now()
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['target'].values)
#        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
             early_stopping_rounds=early_stopping_rounds, verbose_eval=False, metrics='auc')
        alg.set_params(n_estimators=cvresult.shape[0])
        print("cv score:", cvresult.iloc[-1,0])
    
    #fit
    alg.fit(dtrain[predictors], dtrain['target'], eval_metric='auc')
        
    #prediction on train set
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    endtime = datetime.datetime.now()   
    
    #output
    print("accuracy: ", metrics.accuracy_score(dtrain['target'].values, dtrain_predictions))
    print("AUC score:", metrics.roc_auc_score(dtrain['target'], dtrain_predprob))
    print("time spent: ", (endtime - starttime).seconds, "s")  

                

xgb1 = XGBClassifier(
        learning_rate = 0.1,
        n_estimators = 1000,
        max_depth = 5,
        min_child_weight = 1,
        gamma = 0,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = 'binary:logistic',
        nthread = 4,
        scale_pos_weight = 20,                    #deal with imbalaced data
        seed = 666)
   
xgbfit(xgb1, train, predictors, useTrainCV=True)

#cv score: 0.6390654
#accuracy:  0.815942890936
#AUC score: 0.678874017623
#time spent:  3950 s
#'n_estimators': 91

test = pd.read_csv("test.csv")
test = pd.get_dummies(test, columns=cat_cols)
test['target'] = xgb1.predict_proba(test[predictors])[:,1]
test[['id','target']].to_csv('submission.csv', index=False, float_format='%.5f')

#score: 0.273