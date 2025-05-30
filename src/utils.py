import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    report = {}
    for name, model in models.items():
        param_grid = params.get(name, {})
        if param_grid:
            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='r2')
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
        else:
            best_model = model
            best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        score = r2_score(y_test, y_pred)
        report[name] = score
    return report

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

