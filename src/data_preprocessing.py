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

# Generic schema repair utility for score/validation-like files that may be missing some required columns.
def ensure_required_columns(data, required_columns=None, reference_data=None, fill_mode='last'):
    data = data.copy()

    if required_columns is None:
        required_columns = []

    for column in required_columns:
        if column in data.columns:
            continue

        if reference_data is not None and column in reference_data.columns:
            reference_series = reference_data[column].dropna()
            if not reference_series.empty:
                if fill_mode == 'last':
                    data[column] = reference_series.iloc[-1]
                elif fill_mode == 'mean':
                    data[column] = reference_series.mean()
                elif fill_mode == 'median':
                    data[column] = reference_series.median()
                else:
                    data[column] = reference_series.iloc[-1]
                continue

            raise ValueError(f"Reference data contains '{column}' column but there are no non-null reference values available.")

        raise ValueError(f"'{column}' column is missing and no reference_data was provided to fill it.")

    return data

# Backward-compatible convenience helper for the market_index-specific checksum on December files.
def ensure_market_index_column(data, reference_data=None):
    return ensure_required_columns(data, required_columns=['market_index'], reference_data=reference_data)

def preprocess_data(data):
    data = data.copy()

    if 'load_id' in data.columns:
        data = data.drop(columns=['load_id'])

    data = weight_preprocessing(data)
    data = date_preprocessing(data)
    data = market_index_preprocessing(data)
    return data