import pandas as pd
import numpy as np

# Fixing the outlier values
def weight_preprocessing(data):
    data = data.copy()
    data['weight'] = data['weight'].abs()

    data.loc[data['weight'] == 0, 'weight'] = np.nan
    weight_medians_equipment = data.groupby('equipment')['weight'].transform('median')
    data['weight'] = data['weight'].fillna(weight_medians_equipment)

    if data['weight'].isna().any():
        data['weight'] = data['weight'].fillna(data['weight'].median())
    return data

# Fixing the datetime
def date_preprocessing(data):
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date').reset_index(drop=True) # Sort the data by date and reset the index
    return data

# Fixing the market index
def market_index_preprocessing(data):
    data = data.copy()
    data['market_index'] = data['market_index'].interpolate(method='linear') 
    data['market_index'] = data['market_index'].ffill().bfill() 

    return data

def preprocess_data(data):
    data = data.copy()

    if 'load_id' in data.columns:
        data = data.drop(columns=['load_id'])

    data = weight_preprocessing(data)
    data = date_preprocessing(data)
    data = market_index_preprocessing(data)
    return data