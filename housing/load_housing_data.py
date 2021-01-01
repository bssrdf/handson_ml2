# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 22:08:25 2020

@author: merli
"""

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import os
import matplotlib.pyplot as plt

HOUSING_PATH = os.path.join("../datasets", "housing")

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

if __name__ == "__main__":
   housing = load_housing_data()
   print(housing.head())
   print(housing.info())
   print(housing["ocean_proximity"].value_counts())
   print(housing.describe())
   
   
   housing.hist(bins=50, figsize=(20,15))
   #plt.save_fig("attribute_histogram_plots")
   plt.show()