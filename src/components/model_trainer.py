# Basic Import
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'XGBClassifier': XGBClassifier(),
            'GradientBoostingClassifier' : GradientBoostingClassifier(),
            'SVC' : SVC(),
            'RandomForestClassifier': RandomForestClassifier()
        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            # max_acc_model_for_grdsrc=models['best_model_name']
            logging.info('Grid Search CV is going to star')
            parameters={'n_estimators' : [1,2,3,4,5,6,7], 'max_depth' : [1,2,3,4,5,6,7,8]}
            grid_search = GridSearchCV(best_model, param_grid=parameters, cv=5, n_jobs=-1, verbose=3 )
            grid_search.fit(X_train,y_train)
            best_params=grid_search.best_params_
            print("After Grid Search CV")
            print("best params are : ",best_params)
            print('\n====================================================================================\n')
            
            y_test_predicted = grid_search.predict(X_test)
            test_model_score_final = accuracy_score(y_test,y_test_predicted)

            print(f'Perfect Model Found , Model Name : {best_model_name} , Accuracy Score : {test_model_score_final}')

            logging.info(f'Perfect Model Found , Model Name : {best_model_name} , Accuracy Score : {test_model_score_final}')

            
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)




            