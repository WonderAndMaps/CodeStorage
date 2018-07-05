# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:25:32 2017

@author: dell
"""
import os
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
from sklearn import metrics
import datetime


os.chdir('C:\\Users\\dell\\Desktop\\Safe driver')
train = pd.read_csv("train.csv")


y = train['target']
X = train.drop(['id','target'],axis=1)
importance = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)

feature_import = [(a,b) for a,b in zip(X.columns,importance)]
feature_import.sort(key=lambda x:x[1], reverse=True)

#top 40 features
seleceted_predictors = [x[0] for x in feature_import[0:40]]

train = train[seleceted_predictors+['target']]
cat_cols = [col for col in train.columns if '_cat' in col]

#one-hot
train = pd.get_dummies(train, columns=cat_cols)
predictors = [x for x in train.columns if x not in ['target', 'id']]


def lgbfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, dtest=None):
    
    starttime = datetime.datetime.now()
    if useTrainCV:
        lgb_param = alg.get_params()
        ltrain = lgb.Dataset(dtrain[predictors].values, label=dtrain['target'].values)
#        ltest = lgb.Dataset(dtest[predictors].values)
        cvresult = lgb.cv(lgb_param, ltrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
             early_stopping_rounds=early_stopping_rounds, verbose_eval=False, metrics='auc')
        alg.set_params(n_estimators=len(cvresult['auc-mean']))
        print("cv score:", cvresult['auc-mean'][-1])
    
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

lgb1 = LGBMClassifier(
        boosting_type = 'gbdt',
        learning_rate = 0.1,
        n_estimators = 1000,
        max_depth = 5,
        min_child_weight = 1,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = 'binary',
        n_jobs = 4,
        random_state = 66)

lgbfit(lgb1, train, predictors, useTrainCV=True)

#cv score: 0.639641169672
#accuracy:  0.963587763688
#AUC score: 0.68066069449
#time spent:  63 s
#'n_estimators': 107

test = pd.read_csv("test.csv")
test = pd.get_dummies(test, columns=cat_cols)
test['target'] = lgb1.predict_proba(test[predictors])[:,1]
test[['id','target']].to_csv('submission.csv', index=False, float_format='%.5f')

