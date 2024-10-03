import pandas as pd
import numpy as np

# Data
# https://www.kaggle.com/datasets/aramacus/electricity-demand-in-victoria-australia/data

def read_electricity(year_start=2015, year_end=2020):

    # read data
    data = pd.read_csv('data/complete_dataset.csv', thousands=',')

    # impute missing data
    data.at[161, 'rainfall'] = 0
    data.at[1377, 'rainfall'] = 0
    data.at[1378, 'rainfall'] = 0
    data.at[1060, 'solar_exposure'] = (data.loc[1059, 'solar_exposure'] + data.loc[1061, 'solar_exposure']) / 2

    # extract year
    data['year'] = np.array([int(date[:4]) for date in data['date']])

    # filter years in range
    data = data[(data['year'] >= year_start) & (data['year'] <= year_end)]
    
    return data