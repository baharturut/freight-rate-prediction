import warnings
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optuna_hyperparameter_tuning(X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_valid: pd.DataFrame,
                 y_valid: pd.Series,
                 n_trials: int = 20) -> dict:
    
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    def objective(trial):
        params = {
            'objective': 'huber',
            'alpha': 0.9, # huber treshold parameter
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True)   
        }

        model = lgb.LGBMRegressor(**params, n_estimators=1000)
        model.fit(X_train, y_train_log, 
                  eval_set=[(X_valid, y_valid_log)], 
                  callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]),

        predictions_log = model.predict(X_valid)
        predictions_exp = np.clip(np.expm1(predictions_log), a_min=0, a_max=None)  # Ensure no negative predictions
        return np.sqrt(mean_squared_error(y_valid, predictions_exp))

    tune_study = optuna.create_study(direction='minimize')
    tune_study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = tune_study.best_params
    best_params.update({
        'objective': 'huber',
        'alpha': 0.9,
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1
    })

    return best_params

def train_lgbm_model(X_train: pd.DataFrame, 
                     y_train: pd.Series, 
                     X_valid: pd.DataFrame = None, 
                     y_valid: pd.Series = None, 
                     params: dict = None) -> lgb.LGBMRegressor:

    if params is None:
        params = {
            'objective': 'huber',
            'alpha': 0.9,
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': -1,
            'learning_rate': 0.03,
            'num_leaves': 63,
            'max_depth': 8
        }

    y_train_log = np.log1p(y_train)
    model = lgb.LGBMRegressor(**params, n_estimators=2000)

    if X_valid is not None and y_valid is not None:
        y_valid_log = np.log1p(y_valid)
        model.fit(X_train, y_train_log, 
                  eval_set=[(X_valid, y_valid_log)], 
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    else:
        model.fit(X_train, y_train_log)

    return model

def predict_rates(model: lgb.LGBMRegressor, X_test: pd.DataFrame, train_columns: list) -> np.ndarray:
    X_aligned = X_test.reindex(columns=train_columns, fill_value=0)
    preds_log = model.predict(X_aligned)
    preds_exp = np.clip(np.expm1(preds_log), a_min=0, a_max=None)  # Ensure no negative predictions
    return preds_exp

def evaluation(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"Evaluation Metrics:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}")
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }