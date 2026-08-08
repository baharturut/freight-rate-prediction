import pandas as pd

def one_hot_encoding(data: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    data = data.copy()
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    return data

def align_columns(X_target: pd.DataFrame, train_columns: list) -> pd.DataFrame:
   return X_target.reindex(columns=train_columns, fill_value=0)

