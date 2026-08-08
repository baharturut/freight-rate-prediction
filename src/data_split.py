import pandas as pd


def time_based_split(data: pd.DataFrame, target_column: str = 'posted_rate', split_date: str = '2025-11-01'):
    data = data.sort_values('date').reset_index(drop=True)

    train_data_mask = data['date'] < split_date
    val_data_mask = data['date'] >= split_date

    train_data = data[train_data_mask].copy()
    val_data = data[val_data_mask].copy()

    drop_columns = [target_column ,'date']

    X_train = train_data.drop(columns=[c for c in drop_columns if c in train_data.columns])
    y_train = train_data[target_column]

    X_val = val_data.drop(columns=[c for c in drop_columns if c in val_data.columns])
    y_val = val_data[target_column]

    return X_train, y_train, X_val, y_val
