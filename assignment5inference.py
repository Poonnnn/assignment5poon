"""
Inference script for assignment 5.

Loads saved Keras model from `examples/assignment5_model.h5` and scaler,
runs predictions on `data/test.csv`, prints classification metrics and
saves predictions and confusion matrix to output/.
"""

import os
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from sklearn.metrics import classification_report, f1_score, confusion_matrix


def main():
    model_path = 'examples/assignment5_model.h5'
    scaler_path = 'examples/assignment5_scaler.joblib'
    threshold_path = 'examples/assignment5_threshold.joblib'

    if not os.path.exists(model_path):
        raise RuntimeError(f'Model not found: {model_path}. Run training first.')

    if not os.path.exists(scaler_path):
        raise RuntimeError(f'Scaler not found: {scaler_path}. Run training first.')

    # Load model and scaler with custom object
    scaler = joblib.load(scaler_path)
    
    # Load threshold (default to 0.5 if not found)
    threshold = 0.5
    if os.path.exists(threshold_path):
        threshold_dict = joblib.load(threshold_path)
        threshold = threshold_dict.get('threshold', 0.5)
    
    # Load model without loss function compilation (inference only)
    model = keras.models.load_model(model_path, compile=False)

    # load test data
    test_path = 'data/test.csv'
    if not os.path.exists(test_path):
        raise RuntimeError(f'Test file not found: {test_path}')

    df_test = pd.read_csv(test_path)
    if 'ProdTaken' in df_test.columns:
        X_test = df_test.drop(columns=['ProdTaken'])
        y_true = df_test['ProdTaken'].astype(int)
    else:
        X_test = df_test
        y_true = None

    # One-hot encode categorical features to match training
    categorical_cols = X_test.select_dtypes(include=['object']).columns.tolist()
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

    # Align with training features - add missing columns with 0, remove extra columns
    # This handles cases where test set has different categorical values than training
    expected_features = scaler.get_feature_names_out() if hasattr(scaler, 'get_feature_names_out') else None
    
    # Scale the features with error handling
    try:
        X_test_scaled = scaler.transform(X_test_encoded)
    except ValueError:
        # If feature mismatch, try to align columns
        # Reindex to match training features
        if expected_features is not None:
            X_test_encoded = X_test_encoded.reindex(columns=expected_features, fill_value=0)
            X_test_scaled = scaler.transform(X_test_encoded)
        else:
            raise

    # Make predictions with optimal threshold
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    preds = (y_pred_proba > threshold).astype(int).flatten()

    # save predictions
    os.makedirs('output', exist_ok=True)
    out_df = X_test.copy()
    out_df['Pred_Prob'] = y_pred_proba.flatten()
    out_df['Pred'] = preds
    out_df.to_csv('output/predictions.csv', index=False)
    print('Predictions saved to output/predictions.csv')

    if y_true is not None:
        print('\nClassification report on test set:')
        print(classification_report(y_true, preds))
        print(f'F1-score: {f1_score(y_true, preds):.4f}')

        # Confusion matrix (array), save CSV and try to save a heatmap image if plotting libs exist
        cm = confusion_matrix(y_true, preds)
        cm_df = pd.DataFrame(cm, index=['True_0', 'True_1'], columns=['Pred_0', 'Pred_1'])
        cm_df.to_csv('output/tourism_confusingmatrix.csv', index=True)
        print('\nConfusion matrix (saved to output/tourism_confusingmatrix.csv):')
        print(cm_df)

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(5, 4))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig('output/tourism_confusingmatrix.png')
            plt.close()
            print('Confusion matrix image saved to output/tourism_confusingmatrix.png')
        except Exception:
            print('matplotlib/seaborn not available — saved confusion matrix CSV only.')


if __name__ == '__main__':
    main()
