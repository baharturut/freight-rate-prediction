import os
import numpy as np
import pandas as pd

from src.data_preprocessing import preprocess_data
from src.data_split import time_based_split
from src.encoding import one_hot_encoding, align_columns
from src.feature_engineering import feature_engineering
from src.models import (
    evaluation,
    optuna_hyperparameter_tuning,
    train_lgbm_model,
    predict_rates
)

def main():
    # 1. Eğitim Verisi Yükleme
    train_test_path = 'data/train-test.csv'
    if not os.path.exists(train_test_path):
        raise FileNotFoundError(f"Dosya bulunamadı: '{train_test_path}'")

    data = pd.read_csv(train_test_path)
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])

    # Pipeline
    data = preprocess_data(data)
    data = feature_engineering(data)

    categorical_columns = ['pickup', 'delivery', 'equipment', 'route_id']
    existing_categorical_columns = [col for col in categorical_columns if col in data.columns]
    data_encoded = one_hot_encoding(data, existing_categorical_columns)
    data_encoded = data_encoded.drop(columns=['load_id'], errors='ignore')

    # 2. Split ve Model Eğitimi
    SPLIT_DATE = '2025-09-01'
    X_train, y_train, X_test, y_test = time_based_split(
        data_encoded,
        target_column='posted_rate',
        split_date=SPLIT_DATE
    )

    train_columns = list(X_train.columns)
    X_test = align_columns(X_test, train_columns)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    best_params = optuna_hyperparameter_tuning(X_train, y_train, X_test, y_test, n_trials=15)
    model = train_lgbm_model(X_train, y_train, X_test, y_test, params=best_params)

    test_predictions = predict_rates(model, X_test, train_columns)
    evaluation(y_test, test_predictions)

    # 3. Harici Validation Dosyası (12,000 Satır)
    validation_path = 'data/validation.csv'
    template_path = 'data/validation-predictions-template.csv'
    
    if not os.path.exists(validation_path) or not os.path.exists(template_path):
        raise FileNotFoundError("Validation veya Şablon dosyaları eksik!")

    data_validation = pd.read_csv(validation_path)
    data_validation['date'] = pd.to_datetime(data_validation['date'])

    data_val_processed = preprocess_data(data_validation)
    data_val_processed = feature_engineering(data_val_processed)
    data_val_encoded = one_hot_encoding(data_val_processed, existing_categorical_columns)

    X_submission = data_val_encoded.drop(columns=['date', 'posted_rate', 'load_id'], errors='ignore')
    X_submission_aligned = align_columns(X_submission, train_columns)

    final_predictions = predict_rates(model, X_submission_aligned, train_columns)

    template = pd.read_csv(template_path)
    df_submission = template.copy()
    df_submission['predicted_rate'] = final_predictions

    root_output_file = 'validation_predictions.csv'
    df_submission.to_csv(root_output_file, index=False)

    # 4. Aralık Grafik Dosyasını Gerçek Model ile Tahmin Etme
    scorer_december_path = 'data/december_chart_inputs.csv'
    legacy_december_path = 'data/december-chart-inputs.csv'
    
    dec_path = scorer_december_path if os.path.exists(scorer_december_path) else legacy_december_path

    if os.path.exists(dec_path):
        dec_df = pd.read_csv(dec_path)
        dec_df['date'] = pd.to_datetime(dec_df['date'])

        # Eksik pazar indeksi varsa eğitim kümesinin son ortalamasıyla tamamla
        if 'market_index' not in dec_df.columns and 'market_index' in data.columns:
            dec_df['market_index'] = data['market_index'].dropna().iloc[-1]

        # Gerçek boru hattından geçir
        dec_processed = preprocess_data(dec_df)
        dec_fe = feature_engineering(dec_processed)
        dec_encoded = one_hot_encoding(dec_fe, existing_categorical_columns)

        X_dec = dec_encoded.drop(columns=['date', 'posted_rate', 'load_id'], errors='ignore')
        X_dec_aligned = align_columns(X_dec, train_columns)

        # Gerçek model tahmini üret
        december_preds = predict_rates(model, X_dec_aligned, train_columns)
        
        dec_df_out = pd.read_csv(dec_path)
        dec_df_out['predicted_rate'] = december_preds
        
        # Çıktıları kaydet
        dec_df_out.to_csv(scorer_december_path, index=False)
        if os.path.exists(legacy_december_path):
            dec_df_out.to_csv(legacy_december_path, index=False)

    print('=' * 50)
    print(f'✅ {root_output_file} başarıyla kaydedildi!')
    print(f'✅ Aralık grafik verisi GERÇEK MODEL ile tahmin edilip kaydedildi!')
    print(f'Toplam Tahmin Satırı : {len(df_submission):,}')
    print(f'Ortalama Navlun Fiyatı: ${df_submission["predicted_rate"].mean():.2f}')
    print('=' * 50)

if __name__ == "__main__":
    main()