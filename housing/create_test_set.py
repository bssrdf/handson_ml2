# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:11:00 2020

@author: merli
"""

import numpy as np
from zlib import crc32
import hashlib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]



#def test_set_check(identifier, test_ratio):
 #   return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    print(type(ids))
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    print(type(in_test_set))
    return data.loc[~in_test_set], data.loc[in_test_set]

from sklearn.model_selection import StratifiedShuffleSplit
def stratified_sample(data):
    data["income_cat"] = pd.cut(data["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    
    for train_index, test_index in split.split(data, data["income_cat"]):
        
        strat_train_set = data.loc[train_index]
        strat_test_set  = data.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    return strat_train_set, strat_test_set

from load_housing_data import load_housing_data

if __name__ == "__main__":
    np.random.seed(42)
    housing = load_housing_data()
    train_set, test_set = split_train_test(housing, 0.2)
    print(len(train_set))
    print(len(test_set))
    housing_with_id = housing.reset_index()   # adds an `index` column
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    print(len(train_set))
    print(len(test_set))
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
    print(len(train_set))
    print(len(test_set))   

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    print(len(train_set))
    print(len(test_set))
    print(test_set.head())
    
    housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
    print(housing["income_cat"].value_counts())
    housing["income_cat"].hist()
    #plt.show()
    
    from sklearn.model_selection import StratifiedShuffleSplit

    split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
    #print(strat_test_set)
    print(housing["income_cat"].value_counts() / len(housing))
    
    def income_cat_proportions(data):
        return data["income_cat"].value_counts() / len(data)

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
     }).sort_index()
    compare_props["Rand. %error"] = 100 * compare_props["Random"] /  \
                                    compare_props["Overall"] - 100
    compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / \
                                   compare_props["Overall"] - 100
    print(compare_props)
    for set_ in (strat_train_set, strat_test_set):
       set_.drop("income_cat", axis=1, inplace=True)