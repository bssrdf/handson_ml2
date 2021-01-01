# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 20:06:13 2020

@author: merli
"""

import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


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
from prepare_dataset import CombinedAttributesAdder

if __name__ == "__main__":
    
    housing = load_housing_data()
    strat_train_set, strat_test_set = stratified_sample(housing)
    housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])                

    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)    
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
    
    housing_prepared = full_pipeline.fit_transform(housing)   

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

    prepare_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),    
    ('svm_reg', RandomForestRegressor(**grid_search.best_params_))
    ])
    
    prepare_and_predict_pipeline.fit(housing, housing_labels)
    some_data = housing.iloc[:4]
    some_labels = housing_labels.iloc[:4]

    print("Predictions:\t", prepare_and_predict_pipeline.predict(some_data))
    print("Labels:\t\t", list(some_labels))
    
