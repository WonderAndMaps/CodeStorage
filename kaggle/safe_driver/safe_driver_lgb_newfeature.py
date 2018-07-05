# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:41:50 2017

@author: dell
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
import os
import datetime
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier


def gini(y, pred):
    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) -1
    return g

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True

#########################################feature engineering#################
class FeatureBinarizatorAndScaler:
    """ This class needed for scaling and binarization features
    """
    NUMERICAL_FEATURES = list()
    CATEGORICAL_FEATURES = list()
    BIN_FEATURES = list()
    binarizers = dict()
    scalers = dict()

    def __init__(self, numerical=list(), categorical=list(), binfeatures = list(), binarizers=dict(), scalers=dict()):
        self.NUMERICAL_FEATURES = numerical
        self.CATEGORICAL_FEATURES = categorical
        self.BIN_FEATURES = binfeatures
        self.binarizers = binarizers
        self.scalers = scalers

    def fit(self, train_set):
        for feature in train_set.columns:

            if feature.split('_')[-1] == 'cat':
                self.CATEGORICAL_FEATURES.append(feature)
            elif feature.split('_')[-1] != 'bin':
                self.NUMERICAL_FEATURES.append(feature)

            else:
                self.BIN_FEATURES.append(feature)
        for feature in self.NUMERICAL_FEATURES:
            '''StandardScaler implements the Transformer API to compute the mean 
            and standard deviation on a training set so as to be able to later 
            reapply the same transformation on the testing set.
            '''
            scaler = StandardScaler()
            self.scalers[feature] = scaler.fit(np.float64(train_set[feature]).reshape((len(train_set[feature]), 1)))
        for feature in self.CATEGORICAL_FEATURES:
            '''Binarize labels in a one-vs-all fashion (similar to one-hot)
            '''
            binarizer = LabelBinarizer()
            self.binarizers[feature] = binarizer.fit(train_set[feature])


    def transform(self, data):
        binarizedAndScaledFeatures = np.empty((0, 0))
        for feature in self.NUMERICAL_FEATURES:
            if feature == self.NUMERICAL_FEATURES[0]:
                binarizedAndScaledFeatures = self.scalers[feature].transform(np.float64(data[feature]).reshape(
                    (len(data[feature]), 1)))
            else:
                binarizedAndScaledFeatures = np.concatenate((
                    binarizedAndScaledFeatures,
                    self.scalers[feature].transform(np.float64(data[feature]).reshape((len(data[feature]),
                                                                                       1)))), axis=1)
        for feature in self.CATEGORICAL_FEATURES:

            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures,
                                                         self.binarizers[feature].transform(data[feature])), axis=1)

        for feature in self.BIN_FEATURES:
            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures, np.array(data[feature]).reshape((
                len(data[feature]), 1))), axis=1)
        print(binarizedAndScaledFeatures.shape)
        return binarizedAndScaledFeatures
    
    
def preproc(X_train):
    # Adding new features and deleting features with low importance
    multreg = X_train['ps_reg_01'] * X_train['ps_reg_03'] * X_train['ps_reg_02']
    ps_car_reg = X_train['ps_car_13'] * X_train['ps_reg_03'] * X_train['ps_car_13']
    X_train = X_train.drop(['ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',
                            'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12',
                            'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',
                            'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin', 'ps_car_10_cat', 'ps_ind_10_bin',
                            'ps_ind_13_bin', 'ps_ind_12_bin'], axis=1)
    X_train['mult'] = multreg
    X_train['ps_car'] = ps_car_reg
    X_train['ps_ind'] = X_train['ps_ind_03'] * X_train['ps_ind_15']
    return X_train

os.chdir('C:\\Users\\dell\\Desktop\\Safe driver')
X_train = pd.read_csv("train.csv")
y_train = X_train['target']
X_train = X_train.drop(['id', 'target'], axis=1)
X_test = pd.read_csv("test.csv")
test_id = X_test['id'].tolist()
X_test = X_test.drop(['id'], axis=1)
X_train = preproc(X_train)
X_test = preproc(X_test)

binarizerandscaler = FeatureBinarizatorAndScaler()
binarizerandscaler.fit(X_train)
X_train = binarizerandscaler.transform(X_train)
X_test = binarizerandscaler.transform(X_test)


kinetic_train = pd.read_csv('kinetic_feature_train.csv')
X_train = pd.DataFrame(np.concatenate((X_train, np.array(kinetic_train)), axis=1))
kinetic_test = pd.read_csv('kinetic_feature_test.csv')
X_test = pd.DataFrame(np.concatenate((X_test, np.array(kinetic_test)), axis=1))

def lgbfit(alg, X, y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, dtest=None):
    
    starttime = datetime.datetime.now()
    if useTrainCV:
        lgb_param = alg.get_params()
        ltrain = lgb.Dataset(X.values, label=y.values)
#        ltest = lgb.Dataset(dtest[predictors].values)
        cvresult = lgb.cv(lgb_param, ltrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
             early_stopping_rounds=early_stopping_rounds, verbose_eval=False, metrics='auc')
        alg.set_params(n_estimators=len(cvresult['auc-mean']))
        print("cv score:", cvresult['auc-mean'][-1])
    
    #fit
    alg.fit(X, y, eval_metric='auc')
        
    #prediction on train set
    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:,1]
    endtime = datetime.datetime.now()   
    
    #output
    print("accuracy: ", metrics.accuracy_score(y.values, dtrain_predictions))
    print("AUC score:", metrics.roc_auc_score(y, dtrain_predprob))
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

lgbfit(lgb1, X_train, y_train, useTrainCV=True)

test_target = lgb1.predict_proba(X_test)[:,1]
test_target = (np.exp(test_target) - 1.0).clip(0,1).tolist()
sub = pd.DataFrame([test_id,test_target]).transpose()
sub.columns = ['id','target']
sub['id'] = sub['id'].astype('int32')
sub.to_csv('submission.csv', index=False, float_format='%.5f')

###############################Bagging#####################################
bag = BaggingClassifier(lgb1,max_samples=0.8,max_features=0.8)
bag.fit(X_train, y_train)

test_target = bag.predict_proba(X_test)[:,1].tolist()
sub = pd.DataFrame([test_id,test_target]).transpose()
sub.columns = ['id','target']
sub['id'] = sub['id'].astype('int32')
sub.to_csv('submission.csv', index=False, float_format='%.5f')

#################################Bagging tuning##############################