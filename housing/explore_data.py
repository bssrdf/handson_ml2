# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 15:50:45 2020

@author: merli
"""
import os
import matplotlib.pyplot as plt

from setup import save_fig
from setup import IMAGES_PATH as images_path

from load_housing_data import load_housing_data

from create_test_set import stratified_sample

if __name__ == "__main__":
    
    housing = load_housing_data()
    strat_train_set, strat_test_set = stratified_sample(housing)
    
    housing = strat_train_set.copy()

    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    save_fig("bad_visualization_plot")
    
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
    plt.legend()
    save_fig("housing_prices_scatterplot")
    

    import matplotlib.image as mpimg
    filename = "california.png"
    california_img=mpimg.imread(os.path.join(images_path, filename))
    ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                           s=housing['population']/100, label="Population",
                           c="median_house_value", cmap=plt.get_cmap("jet"),
                           colorbar=False, alpha=0.4,
                          )
    plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
               cmap=plt.get_cmap("jet"))
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)
    
    prices = housing["median_house_value"]
    tick_values = np.linspace(prices.min(), prices.max(), 11)
    cbar = plt.colorbar(ticks=tick_values/prices.max())
    cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
    cbar.set_label('Median House Value', fontsize=16)
    
    plt.legend(fontsize=16)
    save_fig("california_housing_prices_plot")
    plt.show()
    
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    
    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"]=housing["population"]/housing["households"]
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
