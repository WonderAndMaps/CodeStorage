# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:01:31 2017

@author: dell
"""
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
import os
import pandas as pd
from sklearn import metrics
import datetime
from scipy.sparse import coo_matrix
from tffm import TFFMClassifier
import tensorflow as tf

os.chdir('C:\\Users\\dell\\Desktop\\Safe driver')
train = pd.read_csv("train.csv")
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
#score: 0.274


'''
#predicted leaves as new features
predLeaf = pd.DataFrame(lgb1.apply(train[predictors]))
predLeaf = pd.get_dummies(predLeaf, columns=predLeaf.columns)
predLeaf = coo_matrix(predLeaf)
predLeaf = predLeaf.tocsr()
target = train['target']
del train


#FM
FM1 = TFFMClassifier(
    order=2,
    rank=5,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    n_epochs=50,
    batch_size=10000,
    init_std=0.001,
    input_type='sparse',
    loss_function=tf.metrics.auc,           #改这里auc
    #源码在C:\\ProgramData\\Anaconda3\\lib\\site-packages
)

FM1.fit(predLeaf, target, show_progress=True)

testLeaf = pd.DataFrame(lgb1.apply(test[predictors]))
testLeaf = pd.get_dummies(testLeaf, columns=testLeaf.columns)
testLeaf = coo_matrix(testLeaf)
testLeaf = testLeaf.tocsr()
test['target'] = FM1.predict_proba(testLeaf)[:,1]
test[['id','target']].to_csv('submission.csv', index=False, float_format='%.5f')
'''