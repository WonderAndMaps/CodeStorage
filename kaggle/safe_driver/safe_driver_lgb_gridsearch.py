# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 22:07:06 2017

@author: dell
"""
from sklearn.grid_search import GridSearchCV 
import datetime


#-----------------------------------------------------num_leaves and min_child_weight
param_test1 = {
 'num_leaves':[x for x in range(20,100,20)],
 'min_child_weight':[x for x in range(1,6,2)]
}
gsearch1 = GridSearchCV(estimator=lgb1, param_grid = param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

starttime = datetime.datetime.now()
gsearch1.fit(train[predictors],train['target'])
endtime = datetime.datetime.now()

print("time spent: ", (endtime - starttime).seconds, "s")
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#{'min_child_weight': 5, 'num_leaves': 20}


param_test2 = {
 'num_leaves':[x for x in range(20,40,5)],
 'min_child_weight':[x for x in range(10,20,2)]
}
gsearch2 = GridSearchCV(estimator=lgb1, param_grid = param_test2, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

starttime = datetime.datetime.now()
gsearch2.fit(train[predictors],train['target'])
endtime = datetime.datetime.now()

print("time spent: ", (endtime - starttime).seconds, "s")
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
#{'min_child_weight': 10, 'num_leaves': 25}


#-----------------------------------------------------min_split_gain
lgb2 = LGBMClassifier(
        boosting_type = 'gbdt',
        learning_rate = 0.1,
        n_estimators = 117,
        min_child_weight = 10,
        num_leaves = 25,
        min_split_gain = 0,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = 'binary',
        n_jobs = 4,
        random_state = 66)

param_test3 = {
 'gamma':[x for x in range(0,20,5)]
}

gsearch3 = GridSearchCV(estimator=lgb2, param_grid = param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

starttime = datetime.datetime.now()
gsearch3.fit(train[predictors],train['target'])
endtime = datetime.datetime.now()

print("time spent: ", (endtime - starttime).seconds, "s")
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
#{'gamma': 0.0}
