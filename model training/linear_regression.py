import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def Logistic_Regression(read_revenue_analysis, read_df):

    # read dataframe columns : x = features, y = target 
    x = read_df[['ATTENDANCE',
                 'ONLINE SALES BEFORE','ONLINE SALES AFTER',
                 'TIKTOK BEFORE', 'GOOGLE BEFORE',
                 'TIKTOK AFTER', 'GOOGLE AFTER']]
    y = read_revenue_analysis['REVENUE INCREASE']

    # split dataset into train & test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(sampling_strategy='minority')),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
     
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
    param_grid = {'clf__C': [0.01, 0.1, 1, 10]}

    clf = GridSearchCV(pipe, param_grid=param_grid,scoring='f1', cv = cv, n_jobs=-1)
    clf = clf.fit(x_train, y_train)
    best = clf.best_estimator_

    y_pred_best = best.predict(x_test)
    accuracy_best = accuracy_score(y_test, y_pred_best)

    features = ['ATTENDANCE',
                'ONLINE SALES BEFORE','ONLINE SALES AFTER',
                'TIKTOK BEFORE', 'GOOGLE BEFORE',
                'TIKTOK AFTER', 'GOOGLE AFTER']

    print(classification_report(y_test, y_pred_best))

    model_coef = best.named_steps['clf'].coef_[0]
    
    # Exponentiate coefficient to get Odds Ratio (since it's Logistic Regression)
    coef_exp = np.exp(model_coef)
    
    print("--- Feature Importance (Odds Ratio) ---")
    for f, c in zip(features, coef_exp):
        print(f"{f} : {round(c,2)}")

    print(f"\n{len(x)} rows analysed\n")
    print(confusion_matrix(y_test, y_pred_best))
    print(f"\nACCURACY : {round(accuracy_best*100,2)}%\n")