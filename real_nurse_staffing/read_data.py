import pandas as pd
import numpy as np

# Data source: daily data from https://a816-health.nyc.gov/hdi/epiquery/visualizations?PageType=ps&PopulationSource=Syndromic
# all files are first converted to CSV (UTF-8) format

def read_ED(path, syndrome, borough, group, year_start=2019, year_end=2023):

    data = pd.read_csv(path + '/data/' + syndrome + '.csv', thousands=',')

    data = data.rename(columns={'Date ':'Date'})

    data.columns = [*data.columns[:-1], 'Count']

    data['year'] = np.array([int(date[-4:]) for date in data['Date']])

    data = data[(data['Dim1Value']==borough) & (data['Dim2Value']==group) & (data['year'] >= year_start) & (data['year'] <= year_end)]
    
    return data['Count'].astype(int), data