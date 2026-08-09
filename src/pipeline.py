import os
import pandas as pd

from src.data_preprocessing import preprocess_data, ensure_market_index_column
from src.data_split import time_based_split
from src.encoding import one_hot_encoding, align_columns
from src.feature_engineering import feature_engineering
from src.models import (
    evaluation,
    optuna_hyperparameter_tuning,
    train_lgbm_model,
    predict_rates
)


CATEGORICAL_COLUMNS = ['pickup', 'delivery', 'equipment', 'route_id']


# This function is used to build the training frame for the model, ensuring that it has the same structure as the training data.
def build_training_frame(raw_data, categorical_columns):
    data = raw_data.copy()

    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])

    data = preprocess_data(data)
    data = feature_engineering(data)

    existing_categorical_columns = [col for col in categorical_columns if col in data.columns]
    data = one_hot_encoding(data, existing_categorical_columns)
    return data.drop(columns=['load_id'], errors='ignore')


# This function is used to build the inference frame for new data, ensuring that it has the same structure as the training data.
def build_inference_frame(raw_data, reference_data, categorical_columns):
    data = raw_data.copy()

    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])

    data = ensure_market_index_column(data, reference_data=reference_data)
    data = preprocess_data(data)
    data = feature_engineering(data)

    existing_categorical_columns = [col for col in categorical_columns if col in data.columns]
    data = one_hot_encoding(data, existing_categorical_columns)
    return data.drop(columns=['date', 'posted_rate', 'load_id'], errors='ignore')


def train_model(train_test_path: str = 'data/train-test.csv', split_date: str = '2025-09-01', categorical_columns=None, n_trials: int = 15):
    if categorical_columns is None:
        categorical_columns = CATEGORICAL_COLUMNS

    if not os.path.exists(train_test_path):
        raise FileNotFoundError(f"train-test.csv not found: '{train_test_path}'")

    raw_data = pd.read_csv(train_test_path)
    if 'date' in raw_data.columns:
        raw_data['date'] = pd.to_datetime(raw_data['date'])

    data_encoded = build_training_frame(raw_data, categorical_columns)

    X_train, y_train, X_test, y_test = time_based_split(
        data_encoded,
        target_column='posted_rate',
        split_date=split_date
    )

    train_columns = list(X_train.columns)
    X_test = align_columns(X_test, train_columns)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    best_params = optuna_hyperparameter_tuning(X_train, y_train, X_test, y_test, n_trials=n_trials)
    model = train_lgbm_model(X_train, y_train, X_test, y_test, params=best_params)

    test_predictions = predict_rates(model, X_test, train_columns)
    evaluation(y_test, test_predictions)

    return {
        'model': model,
        'train_columns': train_columns,
        'reference_data': raw_data,
        'categorical_columns': categorical_columns,
        'y_test': y_test
    }


def generate_validation_predictions(model, train_columns, reference_data, validation_path: str, template_path: str, categorical_columns):
    if not os.path.exists(validation_path) or not os.path.exists(template_path):
        raise FileNotFoundError("Not found: validation.csv or validation-predictions-template.csv")

    data_validation = pd.read_csv(validation_path)

    X_submission = build_inference_frame(data_validation, reference_data, categorical_columns)
    X_submission_aligned = align_columns(X_submission, train_columns)

    final_predictions = predict_rates(model, X_submission_aligned, train_columns)

    template = pd.read_csv(template_path)
    df_submission = template.copy()
    df_submission['predicted_rate'] = final_predictions

    root_output_file = 'validation_predictions.csv'
    df_submission.to_csv(root_output_file, index=False)

    print(f'{root_output_file} is created.')
    print(f'Total Predicted Rows: {len(df_submission):,}')
    print(f'Average Predicted Rate: ${df_submission["predicted_rate"].mean():.2f}')

    return final_predictions


def generate_december_predictions(model, train_columns, reference_data, scorer_december_path: str, categorical_columns):
    dec_path = scorer_december_path

    if not os.path.exists(dec_path):
        raise FileNotFoundError(f"Not found: '{dec_path}'")

    dec_df = pd.read_csv(dec_path)

    X_dec = build_inference_frame(dec_df, reference_data, categorical_columns)
    X_dec_aligned = align_columns(X_dec, train_columns)

    december_preds = predict_rates(model, X_dec_aligned, train_columns)

    dec_df_out = pd.read_csv(dec_path)
    dec_df_out['predicted_rate'] = december_preds

    dec_df_out.to_csv(dec_path, index=False)

    print(f'December predictions are saved to {dec_path}.')
    return december_preds


def run_pipeline(train_test_path='data/train-test.csv', validation_path='data/validation.csv', template_path='data/validation-predictions-template.csv', december_path='data/december_chart_inputs.csv'):
    training_result = train_model(train_test_path=train_test_path)
    model = training_result['model']
    train_columns = training_result['train_columns']
    reference_data = training_result['reference_data']
    categorical_columns = training_result['categorical_columns']

    generate_validation_predictions(model, train_columns, reference_data, validation_path, template_path, categorical_columns)
    generate_december_predictions(model, train_columns, reference_data, december_path, categorical_columns)

    return training_result
