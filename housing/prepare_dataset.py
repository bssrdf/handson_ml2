# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 11:27:02 2020

@author: merli
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from setup import save_fig
from setup import IMAGES_PATH as images_path

from load_housing_data import load_housing_data

from create_test_set import stratified_sample

from sklearn.base import BaseEstimator, TransformerMixin


# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def transform_pipeline(housing):

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
                
    return full_pipeline.fit_transform(housing)                
    

if __name__ == "__main__":
    
    housing = load_housing_data()
    strat_train_set, strat_test_set = stratified_sample(housing)
    housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
    print(sample_incomplete_rows)
    
    # handling missing data
    imputer = SimpleImputer(strategy="median")
    
    housing_num = housing.drop("ocean_proximity", axis=1)
    # alternatively: housing_num = housing.select_dtypes(include=[np.number])
    imputer.fit(housing_num)
    print(imputer.statistics_)
    print(housing_num.median().values)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                             index=housing.index)
    print(housing_tr.loc[sample_incomplete_rows.index.values])
    
    # dealing with categorical attributes    
    housing_cat = housing[["ocean_proximity"]]
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    print(housing_cat_1hot)
    
    col_names = "total_rooms", "total_bedrooms", "population", "households"
    rooms_ix, bedrooms_ix, population_ix, households_ix = \
       [housing.columns.get_loc(c) for c in col_names] # get the column indices

    #Custom transformer
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)
    housing_extra_attribs = pd.DataFrame(housing_extra_attribs,
              columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
              index=housing.index)
    print(housing_extra_attribs.head())
    
    
    housing_prepare = transform_pipeline(housing)
    
    
    # Feature scaling
    # it is important to fit the scalers to
    # the training data only, not to the full dataset (including the test set).
    # Only then can you use them to transform the training set and the
    # test set (and new data).
    