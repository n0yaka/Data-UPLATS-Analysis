import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost_model import train_xgboost

def clean_up_data(filename):

    read = pd.read_excel(filename)

    columns = ['BEFORE BELOW RM1000',
                'BEFORE BELOW RM5000',	
                'BEFORE BELOW RM10,000',
                'BEFORE BELOW RM50,000',
                'AFTER BELOW RM1000',
                'AFTER BELOW RM5000',
                'AFTER BELOW RM10,000',
                'AFTER BELOW RM50,000',]

    empty_row = read[columns].apply(lambda x : x.isna() | (x.astype(str).str.strip()==''))
    cleaned = read[~empty_row.all(axis=1)]

    return cleaned


def count_attendance(read_df):
    
    read_df['ATTENDANCE'] = (read_df[['M1','M2','M3']] == '/').sum(axis=1) >= 3

    return read_df


def analysis_revenue_improvement(read_df): 

    before = ['BEFORE BELOW RM1000',
              'BEFORE BELOW RM5000',
              'BEFORE BELOW RM10,000',
              'BEFORE BELOW RM50,000']
    
    after = ['AFTER BELOW RM1000',
             'AFTER BELOW RM5000',
             'AFTER BELOW RM10,000',
             'AFTER BELOW RM50,000']
    
    inc_results = []

    for _, row in read_df.iterrows():

        rev_before = row[before][row[before] == 1.0].index.tolist()
        rev_after = row[after][row[after] == 1.0].index.tolist()

        bef = rev_before[0] if rev_before else None
        aft = rev_after[0] if rev_after else None

        revenue_column = {
            'BEFORE BELOW RM1000':1000,
            'BEFORE BELOW RM5000':5000,
            'BEFORE BELOW RM10,000':10000,
            'BEFORE BELOW RM50,000':50000,
            'AFTER BELOW RM1000':1000,
            'AFTER BELOW RM5000':5000,
            'AFTER BELOW RM10,000':10000,
            'AFTER BELOW RM50,000':50000}
        
        bef_val = revenue_column.get(bef,0)
        aft_val = revenue_column.get(aft,0)

        if bef_val < aft_val:
            inc_results.append(1.0)
        
        else:
            inc_results.append(0.0)


    read_df['REVENUE INCREASE'] = inc_results

    return read_df 


def analysis_UPLATS_worth(read_df):
    
    before = ['BEFORE BELOW RM1000',
              'BEFORE BELOW RM5000',
              'BEFORE BELOW RM10,000',
              'BEFORE BELOW RM50,000']
    
    after = ['AFTER BELOW RM1000',
             'AFTER BELOW RM5000',
             'AFTER BELOW RM10,000',
             'AFTER BELOW RM50,000']
    
    improve, same, worse, diff = 0, 0, 0, 0
    
    attendance =  read_df[read_df['ATTENDANCE'] >= 3]

    for each, row in attendance.iterrows():
        rev_before = row[before][row[before] == 1.0].index.tolist()
        rev_after = row[after][row[after] == 1.0].index.tolist()

        print(f"{row['NAME']} :")

        bef = ','.join(rev_before).strip("BEFORE BELOW RM").replace(",","")
        print("BEFORE : " + bef)

        aft = ','.join(rev_after).strip("AFTER BELOW RM").replace(",","")
        print("AFTER : " + aft)

        if int(bef) < int(aft):
            diff += int(aft) - int(bef)
            print("IMPROVEMENT\n")
            improve += 1

        elif int(bef) == int(aft):
            print("NO CHANGE\n")
            same += 1

        else:
            diff += int(bef) - int(aft)
            print("WORSEN\n")
            worse += 1
    
    print(f"AVERAGE REVENUE INCREASE : {round(diff / len(attendance),2)}")

    print(f"RESULTS :\nIMPROVEMENTS : {improve}\nNO CHANGE : {same}\nWORSEN : {worse}")




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


def Random_Tree_Classifier(read_revenue_analysis, read_df):

    x = read_df[['ATTENDANCE',
                 'ONLINE SALES BEFORE', 'ONLINE SALES AFTER',
                 'TIKTOK BEFORE', 'GOOGLE BEFORE',
                 'TIKTOK AFTER', 'GOOGLE AFTER']]
    y = read_revenue_analysis['REVENUE INCREASE']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(sampling_strategy='minority')),
        ('clf', RandomForestClassifier(class_weight='balanced'))
    ])

    param_grid = [{
        'clf__n_estimators' : [100,200,500],
        'clf__min_samples_split' : [2, 5, 10],
        'clf__min_samples_leaf' : [1, 2, 4],
        'clf__max_features' : ['sqrt', 'log2', None]
    }]

    clf = GridSearchCV(pipe, param_grid=param_grid,
                       cv = 3,
                       n_jobs=-1,
                       verbose=2,
                       scoring='f1')
    clf = clf.fit(x_train, y_train)
    best = clf.best_estimator_
    y_best_pred = best.predict(x_test)

    best_accuracy = accuracy_score(y_test, y_best_pred)

    features = ['ATTENDANCE',
                'ONLINE SALES BEFORE', 'ONLINE SALES AFTER',
                'TIKTOK BEFORE', 'GOOGLE BEFORE',
                'TIKTOK AFTER', 'GOOGLE AFTER']
    
    print(classification_report(y_test, y_best_pred))
    print(confusion_matrix(y_test, y_best_pred))
    print(f"\nACCURACY : {round(best_accuracy*100,2)}%\n")


def Gradient_Boosting(read_revenue_analysis, read_df): #tak valid

    x = read_df[['ATTENDANCE',
                 'ONLINE SALES BEFORE', 'ONLINE SALES AFTER',
                 'TIKTOK BEFORE', 'GOOGLE BEFORE',
                 'TIKTOK AFTER', 'GOOGLE AFTER']]
    y = read_revenue_analysis['REVENUE INCREASE']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

    model = GradientBoostingClassifier(n_estimators=300,
                                      learning_rate=0.05,
                                      random_state=100)
    
    model.fit(x_train, y_train)
    y_prediction = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_prediction)
    
    print(f"\nACCURACY : {round(accuracy*100, 2)}%\n")
    print(confusion_matrix(y_test, y_prediction))
    print(classification_report(y_test, y_prediction))


def SVM(read_revenue_analysis, read_df):

    x = read_df[['ATTENDANCE','ONLINE SALES AFTER']]
    y = read_revenue_analysis['REVENUE INCREASE']
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    smote = SMOTE(sampling_strategy='minority')
    x_scaled, y = smote.fit_resample(x_scaled, y)

    svm_model = SVC(kernel='linear')
    svm_model.fit(x_scaled, y)

    DecisionBoundaryDisplay.from_estimator(
        svm_model,
        x_scaled,
        response_method='predict',
        alpha=0.8,
        cmap='Pastel1'
    )

    plt.scatter(
        x_scaled[:, 0],
        x_scaled[:, 1],
        c=y,
        s=20,
        edgecolors='k'
    )

    plt.xlabel("Attendence")
    plt.ylabel("Online Sales")

    plt.show()



if __name__ == '__main__':
    file = r"C:\Users\radin\Desktop\VSCODE FILES\Data Analysis\DATA.xlsx"

    cleaned = clean_up_data(file)
    attendance = count_attendance(cleaned)

    revenue = analysis_revenue_improvement(attendance)

    # Logistic_Regression(revenue, attendance)
    # Random_Tree_Classifier(revenue, attendance)
    # Gradient_Boosting(revenue, attendance)
    # SVM(revenue, attendance)
    train_xgboost(revenue, attendance)