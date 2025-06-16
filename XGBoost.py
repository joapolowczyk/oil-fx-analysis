###### XGBOOST ########

# ANALIZA WRAŻLIWOŚCI XGBOOST #

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import matplotlib

matplotlib.use('Agg')

# Konfiguracja ścieżek i parametrów
INPUT_FILE_PATH = "Brent_fxrates.xlsx"
OUTPUT_SENSITIVITY_EXCEL_PATH = r"XGBOOST\xgb_sensitivity.xlsx"

# Parametry analizy
LAGS_BRENT = [1, 2, 3]
CURRENCIES = ["USD/EUR", "USD/CAD", "USD/NOK", "USD/RUB"]
LAGS_CURRENCY = [1, 2, 3]
BRENT_PERTURBATION_PERCENT = 0.01
N_SPLITS_TIMESERIES = 5
N_ESTIMATORS_XGB = 100
RANDOM_STATE_XGB = 42

# Wczytanie i przygotowanie danych
df = pd.read_excel(INPUT_FILE_PATH)
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df.asfreq("D")

# Tworzenie kolumn z lagami dla cen ropy
for lag in LAGS_BRENT:
    df[f"BRENT_lag{lag}"] = df["BRENT"].shift(lag)

#  Główna pętla analizy wrażliwości
results_xgb_sensitivity = []

for waluta in CURRENCIES:

    for lag_brent_to_test in LAGS_BRENT:

        brent_feature_for_this_model = [f"BRENT_lag{lag_brent_to_test}"]
        df_model_xgb = df[[waluta] + brent_feature_for_this_model].copy()
        for lag_waluta in LAGS_CURRENCY:
            df_model_xgb[f"{waluta}_lag{lag_waluta}"] = df_model_xgb[waluta].shift(lag_waluta)

        # W tym miejscu wszystkie wiersze z NaN (z weekendów lub z lagowania) zostaną usunięte
        df_model_xgb.dropna(inplace=True)

        y_xgb = df_model_xgb[waluta]
        X_xgb_cols = brent_feature_for_this_model + [f"{waluta}_lag{lw}" for lw in LAGS_CURRENCY]
        X_xgb = df_model_xgb[X_xgb_cols]

        ts_cv = TimeSeriesSplit(n_splits=N_SPLITS_TIMESERIES)
        fold_sensitivities_plus = []
        fold_sensitivities_minus = []

        for train_index, test_index in ts_cv.split(X_xgb):
            X_train_xgb, X_test_xgb = X_xgb.iloc[train_index], X_xgb.iloc[test_index]
            y_train_xgb, y_test_xgb = y_xgb.iloc[train_index], y_xgb.iloc[test_index]

            if X_train_xgb.empty or y_train_xgb.empty:
                continue

            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=N_ESTIMATORS_XGB,
                                         random_state=RANDOM_STATE_XGB)
            xgb_model.fit(X_train_xgb, y_train_xgb)

            base_predictions = xgb_model.predict(X_test_xgb)

            brent_col_name = f"BRENT_lag{lag_brent_to_test}"

            X_test_plus = X_test_xgb.copy()
            X_test_plus[brent_col_name] = X_test_plus[brent_col_name] * (1 + BRENT_PERTURBATION_PERCENT)
            predictions_plus = xgb_model.predict(X_test_plus)

            X_test_minus = X_test_xgb.copy()
            X_test_minus[brent_col_name] = X_test_minus[brent_col_name] * (1 - BRENT_PERTURBATION_PERCENT)
            predictions_minus = xgb_model.predict(X_test_minus)

            sensitivity_plus = predictions_plus - base_predictions
            sensitivity_minus = predictions_minus - base_predictions

            fold_sensitivities_plus.append(np.mean(sensitivity_plus))
            fold_sensitivities_minus.append(np.mean(sensitivity_minus))

        mean_sens_plus = np.nanmean(fold_sensitivities_plus)
        std_sens_plus = np.nanstd(fold_sensitivities_plus)
        mean_sens_minus = np.nanmean(fold_sensitivities_minus)
        std_sens_minus = np.nanstd(fold_sensitivities_minus)

        results_xgb_sensitivity.append({
            "Waluta": waluta,
            "Lag BRENT": lag_brent_to_test,
            f"Mean Sensitivity (BRENT +{BRENT_PERTURBATION_PERCENT * 100}%)": mean_sens_plus,
            f"Std Sensitivity (BRENT +{BRENT_PERTURBATION_PERCENT * 100}%)": std_sens_plus,
            f"Mean Sensitivity (BRENT -{BRENT_PERTURBATION_PERCENT * 100}%)": mean_sens_minus,
            f"Std Sensitivity (BRENT -{BRENT_PERTURBATION_PERCENT * 100}%)": std_sens_minus,
        })

pd.DataFrame(results_xgb_sensitivity).to_excel(OUTPUT_SENSITIVITY_EXCEL_PATH, index=False)
print(f"\nAnaliza wrażliwości XGBoost zakończona. Wyniki zapisano w: {OUTPUT_SENSITIVITY_EXCEL_PATH}")


# PROGNOZA KURSÓW WALUT #

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.tsa.stattools import acf

matplotlib.use('Agg')

# Konfiguracja ścieżek i parametrów
INPUT_FILE_PATH = "Brent_fxrates.xlsx"
OUTPUT_CHARTS_DIR = r"XGBOOST\charts"
OUTPUT_RESULTS_EXCEL_PATH = r"XGBOOST\xgb_results.xlsx"

# Parametry modelu i analizy
CURRENCIES = ["USD/EUR", "USD/CAD", "USD/NOK", "USD/RUB"]
LAGS_BRENT = [1, 2, 3]
LAGS_CURRENCY = [1, 2, 3]
N_SPLITS_TIMESERIES = 5
START_PLOT_DATE = '2018-01-01'
N_ESTIMATORS_XGB = 100
RANDOM_STATE_XGB = 42

os.makedirs(OUTPUT_CHARTS_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

# Wczytanie i przygotowanie danych
df = pd.read_excel(INPUT_FILE_PATH)
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df.asfreq("D")

# Tworzenie wszystkich cech opóźnionych
df_features = df.copy()
for lag in LAGS_BRENT:
    df_features[f"BRENT_lag{lag}"] = df_features["BRENT"].shift(lag)
for waluta_col in CURRENCIES:
    for lag_w in LAGS_CURRENCY:
        df_features[f"{waluta_col.replace('/', '_')}_lag{lag_w}"] = df_features[waluta_col].shift(lag_w)
df_features.dropna(inplace=True)

# Funkcja do obliczania metryk
def calculate_metrics(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R-kwadrat': r2_score(y_true, y_pred)
    }

# Główna pętla prognozowania
results_xgb = []

for waluta_target in CURRENCIES:
    for brent_lag_val in LAGS_BRENT:
        y = df_features[waluta_target]
        current_features = [f"BRENT_lag{brent_lag_val}"] + [f"{waluta_target.replace('/', '_')}_lag{lw}" for lw in LAGS_CURRENCY]

        if not all(f in df_features.columns for f in current_features):
            continue

        X = df_features[current_features]

        ts_cv = TimeSeriesSplit(n_splits=N_SPLITS_TIMESERIES)
        fold_metrics = []
        fold_feature_importances = []
        all_predictions_for_currency_lag = pd.Series(index=y.index, dtype=float)

        for fold_idx, (train_index, test_index) in enumerate(ts_cv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if X_train.empty:
                continue

            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=N_ESTIMATORS_XGB, random_state=RANDOM_STATE_XGB, n_jobs=-1)
            xgb_model.fit(X_train, y_train)

            y_pred_fold = xgb_model.predict(X_test)
            all_predictions_for_currency_lag.loc[X.index[test_index]] = y_pred_fold

            fold_metrics.append(calculate_metrics(y_test, y_pred_fold))
            fold_feature_importances.append(dict(zip(X_train.columns, xgb_model.feature_importances_)))

        if not fold_metrics:
            results_xgb.append({
                "Waluta": waluta_target, "Lag BRENT": brent_lag_val, "Lags Waluty": LAGS_CURRENCY,
                "Mean MSE": np.nan, "Mean MAE": np.nan, "Mean R-kwadrat": np.nan, "Średnia ważność cech": {}
            })
            continue

        mean_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
        avg_importance = {feat: np.nanmean([imp_dict.get(feat, np.nan) for imp_dict in fold_feature_importances])
                          for feat in X.columns}

        results_xgb.append({
            "Waluta": waluta_target, "Lag BRENT": brent_lag_val, "Lags Waluty": LAGS_CURRENCY,
            **mean_metrics, "Średnia ważność cech": avg_importance
        })

        # Wizualizacja predykcji poziomu
        predictions_to_plot = all_predictions_for_currency_lag[all_predictions_for_currency_lag.index >= START_PLOT_DATE].dropna()
        real_to_plot = y[y.index >= START_PLOT_DATE].reindex(predictions_to_plot.index).dropna()

        if not predictions_to_plot.empty and not real_to_plot.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(real_to_plot.index, real_to_plot, label=f'Rzeczywiste {waluta_target}')
            plt.plot(predictions_to_plot.index, predictions_to_plot, label=f'Prognozy XGBoost (Lag={brent_lag_val})', color='orange')
            plt.xlabel('Data')
            plt.ylabel(f'Kurs {waluta_target}')
            plt.title(f'Prognozy XGBoost dla {waluta_target} (Lag={brent_lag_val})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            safe_currency_name = waluta_target.replace('/', '_')
            lags_currency_str = '_'.join(map(str, LAGS_CURRENCY))
            chart_filename = f"xgb_wykres_fx_lags_cv_{safe_currency_name}_brentlag{brent_lag_val}_walutylags{lags_currency_str}.png"
            plt.savefig(os.path.join(OUTPUT_CHARTS_DIR, chart_filename))
            plt.close()

            # --- Analiza autokorelacji reszt ---
            residuals = real_to_plot - predictions_to_plot
            if not residuals.dropna().empty:
                acf_values = acf(residuals.dropna(), nlags=20, fft=True)
                plt.figure(figsize=(10, 5))
                plt.stem(acf_values)
                plt.title(f'Autokorelacja Reszt (XGBoost, TimeSeriesSplit) - {waluta_target}, Lag BRENT {brent_lag_val}, Lags Waluty {LAGS_CURRENCY}')
                plt.xlabel('Lag')
                plt.ylabel('Autokorelacja')
                plt.grid(True)
                acf_filename = f"acf_residuals_xgb_cv_{safe_currency_name}_brentlag{brent_lag_val}_walutylags{lags_currency_str}.png"
                plt.savefig(os.path.join(OUTPUT_CHARTS_DIR, acf_filename))
                plt.close()

# Zapis wyników do Excela
pd.DataFrame(results_xgb).to_excel(OUTPUT_RESULTS_EXCEL_PATH, index=False)

print(f"Analiza XGBoost z opóźnieniami FX zakończona. Wyniki zapisano w: {OUTPUT_RESULTS_EXCEL_PATH}")
print(f"Wykresy i analizy autokorelacji zapisano w: {OUTPUT_CHARTS_DIR}")