import pandas as pd
from sklearn.metrics import auc, classification_report
import xgboost as xgboost
from sklearn.model_selection import train_test_split

def train_xgboost(read_revenue_analysis, read_df):
    
    x = read_df[[
        'ATTENDANCE',
        'ONLINE SALES BEFORE','ONLINE SALES AFTER',
        'TIKTOK BEFORE','GOOGLE BEFORE',
        'TIKTOK AFTER','GOOGLE AFTER'
    ]]
    y = read_revenue_analysis['REVENUE INCREASE']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

    dtrain_reg = xgboost.DMatrix(x_train, y_train, enable_categorical=True)
    dtest_reg = xgboost.DMatrix(x_test, y_test, enable_categorical=True)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() #sbb imbalance data mmg pakai formula ni

    params = {
        'objective' : 'binary:logistic',
        'tree_method' : 'hist',
        'scale_pos_weight' : scale_pos_weight,
        'eval_metric' : ['auc', 'logloss'],
    }

    evals = [(dtrain_reg, 'train'),(dtest_reg, 'test')]

    #train
    clf = xgboost.train(
        params = params,
        dtrain = dtrain_reg,
        evals = evals,
        verbose_eval = 10,
        num_boost_round = 100
    )

    y_prediction = clf.predict(dtest_reg)

    print(classification_report(y_test, y_prediction.round()))
    print(f"\nACCURACY : {round((y_test == y_prediction.round()).mean()*100,2)}%\n")