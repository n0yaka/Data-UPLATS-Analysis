from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline


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