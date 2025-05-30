import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass  
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessing_path):
        try:
           logging.info("Splitting Dependent and Independent variables from train and test data")
           X_train, y_train = train_array[:, :-1], train_array[:, -1]
           X_test, y_test = test_array[:, :-1], test_array[:, -1]
           models = {
               "Random Forest": RandomForestRegressor(),
               "Decision Tree": DecisionTreeRegressor(),
               "Gradient Boosting": GradientBoostingRegressor(),
               "Linear Regression": LinearRegression(),
               "K-Neighbors Classifier": KNeighborsRegressor(),
               "XGBClassifier": XGBRegressor(),
               "CatBoosting Regressor": CatBoostRegressor(verbose=False),
               "AdaBoost Regressor": AdaBoostRegressor(),
               } 
           
           params={
               "Decision Tree": {
                   'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                  # 'splitter':['best','random'],
                   #'max_features':['sqrt','log2'],
                   'max_depth':[None, 10, 20, 30, 40, 50],}, 

               "Random Forest":{
                   'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                  #'max_features':['sqrt','log2',None],
                   'max_depth':[None, 10, 20, 30, 40, 50],
                   'n_estimators':[50,100,200]
                   },

               "Gradient Boosting": {
                   #'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                   'learning_rate':[0.1,0.01,0.05,0.001],
                   'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]
                   },

               "Linear Regression":{},
               "K-Neighbors Classifier": {
                   'n_neighbors': [3, 5, 7, 9],
                   'weights': ['uniform', 'distance'],
                   'metric': ['minkowski', 'euclidean', 'manhattan']
               },

               "XGBClassifier":{
                   'learning_rate':[0.1,0.01,0.05,0.001],
                  'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]
                  },

               "CatBoosting Regressor": {
                   'depth': [6, 8, 10],
                   'learning_rate': [0.01, 0.05, 0.1],
                   'iterations': [30, 50, 100]
               },

               "AdaBoost Regressor": {
                   'learning_rate': [0.01, 0.05, 0.1],
                   'n_estimators': [50, 100, 200]
               }

           }


           model_report: dict = evaluate_models(
               X_train=X_train,
               y_train=y_train,
               X_test=X_test,
               y_test=y_test,
               models=models,
               params=params  # <-- corrected argument name
           )

           #To get best model score from dict
           best_model_score = max(sorted(model_report.values()))

           # To get best model name from dict

           best_model_name = list(model_report.keys())[
               list(model_report.values()).index(best_model_score)
           ]
           best_model = models[best_model_name]

           if best_model_score<0.6:
               raise CustomException("No best model found")
           
           logging.info(f"Best found model on both training and testing dataset")   

           save_object(
               file_path=self.model_trainer_config.trained_model_file_path,
               obj=best_model  
           ) 

           predicted=best_model.predict(X_test)
           r2_square = r2_score(y_test, predicted)
           return r2_square
        except Exception as e:
            raise CustomException(e,sys)