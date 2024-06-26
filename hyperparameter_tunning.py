from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

def objective_log(trial,X_train, y_train, X_val, y_val):
    C= trial.suggest_float('C',1e-6,1e2,log=True)
    max_iter= trial.suggest_int('max_iter',2000,3000)
    
    model= LogisticRegression(solver='lbfgs', max_iter= max_iter,C= C,class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred= model.predict(X_val)
    accuracy= accuracy_score(y_val, y_pred)
    
    return accuracy

def objective_rf(trial, X_train, y_train, X_val, y_val):
    n_estimators= trial.suggest_int('n_estimators', 10,200)
    max_depth= trial.suggest_int('max_depth' ,2, 32)
    min_sample_split= trial.suggest_int('min_samples_split',2,16)
    min_sample_leaf= trial.suggest_int('min_samples_leaf',1,16)
    
    model= RandomForestClassifier(
        n_estimators= n_estimators,
        max_depth= max_depth,
        min_samples_split= min_sample_split,
        min_samples_leaf= min_sample_leaf,
        random_state= 0,
        class_weight= 'balanced' 
    )
    model.fit(X_train, y_train)
    y_pred= model.predict(X_val)
    accuracy= accuracy_score(y_val, y_pred)
    
    return accuracy

def objective_xgb(trial,X_train, y_train_enc, X_val, y_val_enc):
    n_estimators= trial.suggest_int('n_estimators', 3,200)
    learning_rate= trial.suggest_categorical('learning_rate' ,np.logspace(-4,-1,num=20))
    max_depth= trial.suggest_int('max_depth',2,16)
    
    model= XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_estimators= n_estimators,
        max_depth= max_depth,
        learning_rate= learning_rate,
        random_state= 0,
    )
    model.fit(X_train, y_train_enc)
    y_pred= model.predict(X_val)
    accuracy= accuracy_score(y_val_enc, y_pred)
    
    return accuracy