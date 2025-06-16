###### LSTM #####

# ANALIZA WRAŻLIWOŚCI LSTM #

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings

# Ignoruj ostrzeżenia z TensorFlow
warnings.filterwarnings("ignore", category=UserWarning)

# Konfiguracja parametrów
INPUT_FILE_PATH = "Brent_fxrates.xlsx"
OUTPUT_SENSITIVITY_EXCEL_PATH = r"LSTM\lstm_sensitivity.xlsx"

CURRENCIES = ["USD/EUR", "USD/CAD", "USD/NOK", "USD/RUB"]
LAGS_BRENT = [1, 2, 3]
LAGS_CURRENCY = [1, 2, 3]
LOOK_BACK = 5
BRENT_PERTURBATION_PERCENT = 0.01
N_SPLITS_TIMESERIES = 5

# Parametry modelu LSTM
LSTM_UNITS = 50
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 3

# KROK 1: Przygotowanie danych
df_raw = pd.read_excel(INPUT_FILE_PATH)
df_raw.columns = df_raw.columns.str.strip()
df_raw["Date"] = pd.to_datetime(df_raw["Date"])
df_raw.set_index("Date", inplace=True)
df_raw = df_raw.asfreq("D").ffill().bfill()

for lag in LAGS_BRENT:
    df_raw[f"BRENT_lag{lag}"] = df_raw["BRENT"].shift(lag)
for waluta_col in CURRENCIES:
    for lag in LAGS_CURRENCY:
        df_raw[f"{waluta_col}_lag{lag}"] = df_raw[waluta_col].shift(lag)
df_raw.dropna(inplace=True)


def create_sequences(X, y, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:(i + look_back)])
        ys.append(y[i + look_back])
    return np.array(Xs), np.array(ys)


# Funkcja do budowy i kompilacji modelu LSTM
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(LSTM_UNITS, activation='tanh', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# Główna pętla analizy wrażliwości LSTM
results_lstm_sensitivity = {}

for waluta in CURRENCIES:
    results_lstm_sensitivity[waluta] = {}

    for lag_brent_to_test in LAGS_BRENT:

        brent_feature_for_this_model = [f"BRENT_lag{lag_brent_to_test}"]
        currency_features = [f"{waluta}_lag{lag}" for lag in LAGS_CURRENCY]
        features = brent_feature_for_this_model + currency_features
        target_col = waluta

        X_data_full = df_raw[features].values
        y_data_full = df_raw[target_col].values.reshape(-1, 1)

        ts_cv = TimeSeriesSplit(n_splits=N_SPLITS_TIMESERIES)
        fold_sensitivities_plus = []
        fold_sensitivities_minus = []

        for fold, (train_idx, test_idx) in enumerate(ts_cv.split(X_data_full)):
            X_train_raw, X_test_raw = X_data_full[train_idx], X_data_full[test_idx]
            y_train_raw, y_test_raw = y_data_full[train_idx], y_data_full[test_idx]

            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_train_scaled = scaler_X.fit_transform(X_train_raw)
            y_train_scaled = scaler_y.fit_transform(y_train_raw)

            X_test_scaled_base = scaler_X.transform(X_test_raw)

            X_test_raw_plus = X_test_raw.copy()
            X_test_raw_plus[:, 0] = X_test_raw_plus[:, 0] * (1 + BRENT_PERTURBATION_PERCENT)
            X_test_scaled_plus = scaler_X.transform(X_test_raw_plus)

            X_test_raw_minus = X_test_raw.copy()
            X_test_raw_minus[:, 0] = X_test_raw_minus[:, 0] * (1 - BRENT_PERTURBATION_PERCENT)
            X_test_scaled_minus = scaler_X.transform(X_test_raw_minus)

            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, LOOK_BACK)

            if len(X_train_seq) == 0 or len(X_test_scaled_base) <= LOOK_BACK:
                continue

            X_test_seq_base, y_test_seq_base = create_sequences(X_test_scaled_base, scaler_y.transform(y_test_raw),
                                                                LOOK_BACK)
            X_test_seq_plus, _ = create_sequences(X_test_scaled_plus, scaler_y.transform(y_test_raw), LOOK_BACK)
            X_test_seq_minus, _ = create_sequences(X_test_scaled_minus, scaler_y.transform(y_test_raw), LOOK_BACK)

            model = build_lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE,
                                           restore_best_weights=True)

            model.fit(X_train_seq, y_train_seq, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, validation_split=0.1,
                      verbose=0, callbacks=[early_stopping])

            pred_scaled_base = model.predict(X_test_seq_base, verbose=0)
            pred_scaled_plus = model.predict(X_test_seq_plus, verbose=0)
            pred_scaled_minus = model.predict(X_test_seq_minus, verbose=0)

            pred_original_base = scaler_y.inverse_transform(pred_scaled_base)
            pred_original_plus = scaler_y.inverse_transform(pred_scaled_plus)
            pred_original_minus = scaler_y.inverse_transform(pred_scaled_minus)

            sensitivity_plus = pred_original_plus - pred_original_base
            sensitivity_minus = pred_original_minus - pred_original_base

            fold_sensitivities_plus.append(np.mean(sensitivity_plus))
            fold_sensitivities_minus.append(np.mean(sensitivity_minus))

        mean_sens_plus = np.nanmean(fold_sensitivities_plus)
        std_sens_plus = np.nanstd(fold_sensitivities_plus)
        mean_sens_minus = np.nanmean(fold_sensitivities_minus)
        std_sens_minus = np.nanstd(fold_sensitivities_minus)

        results_lstm_sensitivity[waluta][lag_brent_to_test] = {
            f"Mean Sensitivity (BRENT +{BRENT_PERTURBATION_PERCENT * 100}%)": mean_sens_plus,
            f"Std Sensitivity (BRENT +{BRENT_PERTURBATION_PERCENT * 100}%)": std_sens_plus,
            f"Mean Sensitivity (BRENT -{BRENT_PERTURBATION_PERCENT * 100}%)": mean_sens_minus,
            f"Std Sensitivity (BRENT -{BRENT_PERTURBATION_PERCENT * 100}%)": std_sens_minus,
        }

with pd.ExcelWriter(OUTPUT_SENSITIVITY_EXCEL_PATH) as writer:
    for waluta, lag_results in results_lstm_sensitivity.items():
        df_sensitivity = pd.DataFrame.from_dict(lag_results, orient='index')
        df_sensitivity.index.name = 'Lag BRENT'
        sheet_name = f'Sensitivity_{waluta.replace("/", "_")}'
        df_sensitivity.to_excel(writer, sheet_name=sheet_name)

print(f"\nAnaliza wrażliwości LSTM zakończona. Wyniki zapisano w: {OUTPUT_SENSITIVITY_EXCEL_PATH}")

# PROGNOZA KURSÓW WALUT LSTM #

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
import warnings
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random

# ustawienie ziarna losowości w celu powtarzalności wyników
SEED = 22
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use('Agg')

# Konfiguracja parametrów
FILE_PATH = "Brent_fxrates.xlsx"
OUTPUT_CHARTS_PATH = r"LSTM\charts"
OUTPUT_RESULTS_EXCEL_PATH = r"LSTM\wyniki_lstm.xlsx"

CURRENCIES = ["USD/EUR", "USD/CAD", "USD/NOK", "USD/RUB"]
LOOK_BACK = 30
EPOCHS = 50
BATCH_SIZE = 32
LAGS_BRENT = [1, 2, 3]
LAGS_CURRENCY = [1, 2, 3]
N_SPLITS = 5
START_PLOT_DATE = '2018-01-01'
VALIDATION_SET_SIZE = 0.1  # 10% danych treningowych na zbiór walidacyjny
EARLY_STOPPING_PATIENCE = 5

# Przygotowanie danych
df = pd.read_excel(FILE_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df.columns = df.columns.str.strip()
df = df.asfreq("D").ffill().bfill()

for lag in LAGS_BRENT:
    df[f'BRENT_lag{lag}'] = df['BRENT'].shift(lag)
for currency_col in CURRENCIES:
    for lag_w in LAGS_CURRENCY:
        df[f'{currency_col}_lag{lag_w}'] = df[currency_col].shift(lag_w)
df.dropna(inplace=True)


def create_sequences(X_data, y_data, look_back):
    X, y = [], []
    for i in range(len(X_data) - look_back):
        X.append(X_data[i:(i + look_back), :])
        y.append(y_data[i + look_back])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='tanh', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


results_lstm_metrics = []
os.makedirs(OUTPUT_CHARTS_PATH, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

for currency_org in CURRENCIES:
    currency_safe = currency_org.replace('/', '_')

    for lag_brent_val in LAGS_BRENT:

        brent_feature = [f'BRENT_lag{lag_brent_val}']
        currency_features = [f'{currency_org}_lag{lag}' for lag in LAGS_CURRENCY]
        features = brent_feature + currency_features
        target_col = currency_org

        df_model_ready = df[[target_col] + features].copy()

        X_data = df_model_ready[features].values
        y_data = df_model_ready[target_col].values.reshape(-1, 1)

        ts_cv = TimeSeriesSplit(n_splits=N_SPLITS)
        fold_results = []
        prediction_dates = []
        prediction_values = []

        for fold_idx, (train_idx, test_idx) in enumerate(ts_cv.split(X_data)):
            X_train_raw, X_test_raw = X_data[train_idx], X_data[test_idx]
            y_train_raw, y_test_raw = y_data[train_idx], y_data[test_idx]

            if len(X_train_raw) <= LOOK_BACK or len(X_test_raw) <= LOOK_BACK: continue

            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_train_scaled = scaler_X.fit_transform(X_train_raw)
            y_train_scaled = scaler_y.fit_transform(y_train_raw)
            X_test_scaled = scaler_X.transform(X_test_raw)

            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, LOOK_BACK)
            X_test_seq, y_test_seq = create_sequences(X_test_scaled, scaler_y.transform(y_test_raw), LOOK_BACK)

            if X_train_seq.shape[0] == 0: continue

            model = build_lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))

            early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE,
                                           restore_best_weights=True)

            model.fit(X_train_seq, y_train_seq, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_split=VALIDATION_SET_SIZE,
                      verbose=0,
                      callbacks=[early_stopping])

            # Predykcja na zbiorze testowym
            y_pred_test_scaled = model.predict(X_test_seq, verbose=0)

            # Odwrócenie skalowania
            y_pred_test_original = scaler_y.inverse_transform(y_pred_test_scaled)
            y_test_original = scaler_y.inverse_transform(y_test_seq)

            test_dates = df_model_ready.index[test_idx][LOOK_BACK:]
            prediction_dates.extend(test_dates)
            prediction_values.extend(y_pred_test_original.flatten())

            fold_results.append({
                'MSE': mean_squared_error(y_test_original, y_pred_test_original),
                'MAE': mean_absolute_error(y_test_original, y_pred_test_original),
                'R-kwadrat': r2_score(y_test_original, y_pred_test_original)
            })

        if not fold_results: continue

        results_lstm_metrics.append({
            "Waluta": currency_org, "Lag BRENT": lag_brent_val,
            "Mean MSE": np.nanmean([res['MSE'] for res in fold_results]),
            "Mean MAE": np.nanmean([res['MAE'] for res in fold_results]),
            "Mean R-kwadrat": np.nanmean([res['R-kwadrat'] for res in fold_results])
        })

        all_predictions_agg = pd.Series(data=prediction_values, index=prediction_dates).sort_index()
        predictions_to_plot = all_predictions_agg[all_predictions_agg.index >= START_PLOT_DATE].dropna()
        real_to_plot = df[currency_org].loc[predictions_to_plot.index]

        if not predictions_to_plot.empty:
            plt.figure(figsize=(15, 7))
            plt.plot(real_to_plot.index, real_to_plot, label=f'Rzeczywiste {currency_org}')
            plt.plot(predictions_to_plot.index, predictions_to_plot, label=f'Prognozy LSTM (Lag={lag_brent_val})', color='orange')
            plt.title(f'Prognozy LSTM dla {currency_org} (Lag={lag_brent_val})')
            plt.xlabel('Data')
            plt.ylabel(f'Kurs {currency_org}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            filename = os.path.join(OUTPUT_CHARTS_PATH, f'lstm_forecast_brentlag{lag_brent_val}_{currency_safe}.png')
            plt.savefig(filename)
            plt.close()

results_df_lstm = pd.DataFrame(results_lstm_metrics)
results_df_lstm.to_excel(OUTPUT_RESULTS_EXCEL_PATH, index=False)

print(f"\nWyniki oceny LSTM zostały zapisane do: {OUTPUT_RESULTS_EXCEL_PATH}")
print(f"Wykresy zostały zapisane w: {OUTPUT_CHARTS_PATH}")