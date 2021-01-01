# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 21:16:28 2020

@author: merli
"""

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from load_housing_data import load_housing_data
from create_test_set import stratified_sample
from prepare_dataset import transform_pipeline

if __name__ == "__main__":
    
    housing = load_housing_data()
    strat_train_set, strat_test_set = stratified_sample(housing)
    housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()


    housing_prepared = transform_pipeline(housing) 
    
    '''
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)    

    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    print(lin_mae)        

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print(tree_rmse)
    
    
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    def display_scores(scores):
       print("Scores:", scores)
       print("Mean:", scores.mean())
       print("Standard deviation:", scores.std())

    display_scores(tree_rmse_scores)
    
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)
    
      

    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(housing_prepared, housing_labels)
    forest_mse = mean_squared_error(housing_labels, housing_predictions)
    forest_rmse = np.sqrt(forest_mse)
    print(forest_rmse)
    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    display_scores(forest_rmse_scores)    
    '''
       

    param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)    
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)
    
    
    final_model = grid_search.best_estimator_

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_prepared = transform_pipeline(X_test)
    final_predictions = final_model.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)