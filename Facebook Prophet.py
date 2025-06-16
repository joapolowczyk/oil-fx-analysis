########## FACEBOOK PROPHET #########

# ANALIZA WRAŻLIWOŚCI CEN ROPY NA KURSY WALUT #

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
import warnings

# Ignoruj przyszłe ostrzeżenia z Prophet
warnings.filterwarnings("ignore", category=FutureWarning)

# Konfiguracja parametrów
FILE_PATH = "Brent_fxrates.xlsx"
OUTPUT_SENSITIVITY_EXCEL_PATH = r"FP\prophet_sensitivity.xlsx"

CURRENCIES = ["USD/EUR", "USD/CAD", "USD/NOK", "USD/RUB"]
LAGS_BRENT = [1, 2, 3]
LAGS_CURRENCY = [1, 2, 3]
BRENT_PERTURBATION_PERCENT = 0.01
N_SPLITS_TIMESERIES = 5

#KROK 1: Przygotowanie danych

df_raw = pd.read_excel(FILE_PATH)
df_raw.columns = df_raw.columns.str.strip()
df_raw["Date"] = pd.to_datetime(df_raw["Date"])
df_raw.set_index("Date", inplace=True)
df_raw = df_raw.asfreq("D")

for lag in LAGS_BRENT:
    df_raw[f'brent_lag_{lag}'] = df_raw['BRENT'].shift(lag)

# Główna pętla analizy wrażliwości
results_prophet_sensitivity = {}

for waluta in CURRENCIES:
    results_prophet_sensitivity[waluta] = {}

    for lag_brent_to_test in LAGS_BRENT:

        brent_regressor_for_this_model = [f'brent_lag_{lag_brent_to_test}']
        df_prophet = df_raw[[waluta] + brent_regressor_for_this_model].copy()
        df_prophet.rename(columns={waluta: 'y'}, inplace=True)
        df_prophet['ds'] = df_prophet.index

        for lag_w in LAGS_CURRENCY:
            df_prophet[f'{waluta}_lag_{lag_w}'] = df_prophet['y'].shift(lag_w)

        df_prophet_prepared = df_prophet.dropna()
        REGRESSORS = brent_regressor_for_this_model + [f'{waluta}_lag_{lw}' for lw in LAGS_CURRENCY]

        ts_cv = TimeSeriesSplit(n_splits=N_SPLITS_TIMESERIES)

        fold_sensitivities_plus = []
        fold_sensitivities_minus = []

        for fold, (train_index, test_index) in enumerate(ts_cv.split(df_prophet_prepared)):
            train_df_fold = df_prophet_prepared.iloc[train_index]
            test_df_fold = df_prophet_prepared.iloc[test_index]

            if train_df_fold.empty or test_df_fold.empty:
                continue

            model = Prophet(seasonality_mode='multiplicative', daily_seasonality=False)
            for regressor in REGRESSORS:
                model.add_regressor(regressor)
            model.fit(train_df_fold)

            future_base = test_df_fold.copy()
            future_plus = test_df_fold.copy()
            future_minus = test_df_fold.copy()

            brent_col_name = f'brent_lag_{lag_brent_to_test}'
            future_plus[brent_col_name] = future_plus[brent_col_name] * (1 + BRENT_PERTURBATION_PERCENT)
            future_minus[brent_col_name] = future_minus[brent_col_name] * (1 - BRENT_PERTURBATION_PERCENT)

            forecast_base = model.predict(future_base[['ds'] + REGRESSORS])
            forecast_plus = model.predict(future_plus[['ds'] + REGRESSORS])
            forecast_minus = model.predict(future_minus[['ds'] + REGRESSORS])

            results_df = pd.DataFrame({
                'yhat_base': forecast_base['yhat'],
                'yhat_plus': forecast_plus['yhat'],
                'yhat_minus': forecast_minus['yhat']
            })

            results_df['sensitivity_plus'] = results_df['yhat_plus'] - results_df['yhat_base']
            results_df['sensitivity_minus'] = results_df['yhat_minus'] - results_df['yhat_base']

            fold_sensitivities_plus.append(results_df['sensitivity_plus'].mean())
            fold_sensitivities_minus.append(results_df['sensitivity_minus'].mean())

        mean_sens_plus = np.nanmean(fold_sensitivities_plus)
        std_sens_plus = np.nanstd(fold_sensitivities_plus)
        mean_sens_minus = np.nanmean(fold_sensitivities_minus)
        std_sens_minus = np.nanstd(fold_sensitivities_minus)

        results_prophet_sensitivity[waluta][lag_brent_to_test] = {
            f"Mean Sensitivity (BRENT +{BRENT_PERTURBATION_PERCENT * 100}%)": mean_sens_plus,
            f"Std Sensitivity (BRENT +{BRENT_PERTURBATION_PERCENT * 100}%)": std_sens_plus,
            f"Mean Sensitivity (BRENT -{BRENT_PERTURBATION_PERCENT * 100}%)": mean_sens_minus,
            f"Std Sensitivity (BRENT -{BRENT_PERTURBATION_PERCENT * 100}%)": std_sens_minus,
        }

# Zapis wyników do Excela
with pd.ExcelWriter(OUTPUT_SENSITIVITY_EXCEL_PATH) as writer:
    for waluta, lag_results in results_prophet_sensitivity.items():
        df_to_save = pd.DataFrame.from_dict(lag_results, orient='index')
        df_to_save.index.name = 'Lag BRENT'
        sheet_name = f'Sensitivity_{waluta.replace("/", "_")}'
        df_to_save.to_excel(writer, sheet_name=sheet_name)

print(f"\nAnaliza wrażliwości zakończona. Wyniki zapisano w: {OUTPUT_SENSITIVITY_EXCEL_PATH}")


# PROGNOZA KURSÓW WALUT #

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.use('Agg')

# Konfiguracja parametrów
FILE_PATH = "Brent_fxrates.xlsx"
OUTPUT_CHARTS_PATH = r"FP\charts"
OUTPUT_RESULTS_EXCEL_PATH = r"FP\prophet_evaluation_metrics.xlsx"

CURRENCIES = ["USD/EUR", "USD/CAD", "USD/NOK", "USD/RUB"]
LAGS_BRENT = [1, 2, 3]
LAGS_CURRENCY = [1, 2, 3]
N_SPLITS_TIMESERIES = 5
START_PLOT_DATE = '2018-01-01'

# Przygotowanie danych (globalne)
df = pd.read_excel(FILE_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df.columns = df.columns.str.strip()
df = df.asfreq("D")

# Dodaj kolumny z lagami BRENT globalnie
for lag in LAGS_BRENT:
    df[f'BRENT_lag_{lag}'] = df['BRENT'].shift(lag)

# Dodaj kolumny z lagami walut globalnie
for currency_col in CURRENCIES:
    for lag_w in LAGS_CURRENCY:
        safe_col_name = currency_col.replace('/', '_')
        df[f'{safe_col_name}_lag_{lag_w}'] = df[currency_col].shift(lag_w)


# Funkcje pomocnicze
def get_prophet_df(base_df, target_currency, brent_lag_val, currency_lags_list):
    """
    Przygotowuje DataFrame w formacie wymagany przez Prophet (ds, y)
    oraz dodaje wybrane regresory.
    """
    regressors = [f'BRENT_lag_{brent_lag_val}'] + \
                 [f'{target_currency.replace("/", "_")}_lag_{lw}' for lw in currency_lags_list]

    # Upewnij się, że wszystkie kolumny potrzebne są w df_temp
    cols_to_select = [target_currency] + regressors
    df_temp = base_df[cols_to_select].dropna().reset_index()
    df_temp.rename(columns={'Date': 'ds', target_currency: 'y'}, inplace=True)
    return df_temp, regressors


def train_and_predict_prophet(train_df, test_df, regressors):
    """Trenuje model Prophet i dokonuje predykcji."""
    model = Prophet()
    for reg in regressors:
        model.add_regressor(reg)
    model.fit(train_df)
    forecast = model.predict(test_df[['ds'] + regressors])
    return forecast


def calculate_metrics(y_true, y_pred):
    """Oblicza MSE, MAE i R-kwadrat."""
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R-kwadrat': r2_score(y_true, y_pred)
    }


# Główna pętla oceny modelu
results_prophet_metrics = []
os.makedirs(OUTPUT_CHARTS_PATH, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

print("\nRozpoczynanie oceny modelu Prophet z TimeSeriesSplit...")

for target_currency in CURRENCIES:
    for brent_lag_val in LAGS_BRENT:
        # Przygotuj dane dla bieżącej waluty i laga BRENT
        df_prophet_ready, current_regressors = get_prophet_df(df, target_currency, brent_lag_val, LAGS_CURRENCY)

        if df_prophet_ready.empty:
            print(f"Brak wystarczających danych dla {target_currency} z lagiem BRENT {brent_lag_val}. Pomijam.")
            results_prophet_metrics.append({
                "Waluta": target_currency, "Lag BRENT": brent_lag_val, "Lags Waluty": LAGS_CURRENCY,
                "Mean MSE": np.nan, "Mean MAE": np.nan, "Mean R-kwadrat": np.nan
            })
            continue

        all_predictions = pd.Series(index=df_prophet_ready['ds'], dtype=float)

        ts_cv = TimeSeriesSplit(n_splits=N_SPLITS_TIMESERIES)
        fold_results = []

        for fold_idx, (train_index, test_index) in enumerate(ts_cv.split(df_prophet_ready)):
            train_fold = df_prophet_ready.iloc[train_index].copy()
            test_fold = df_prophet_ready.iloc[test_index].copy()

            if train_fold.empty or test_fold.empty:
                print(f"  Pusty fold treningowy/testowy ({fold_idx + 1}/{N_SPLITS_TIMESERIES}) dla {target_currency}, lag {brent_lag_val}. Pomijam.")
                continue

            # Trening i predykcja
            forecast_fold = train_and_predict_prophet(train_fold, test_fold, current_regressors)

            # Ocena metryk
            metrics = calculate_metrics(test_fold['y'], forecast_fold['yhat'])
            fold_results.append(metrics)

            # Agregacja prognoz

            all_predictions.loc[forecast_fold['ds']] = forecast_fold['yhat'].values

        # Średnie wyniki z walidacji krzyżowej
        if not fold_results:
            mean_mse, mean_mae, mean_r2 = np.nan, np.nan, np.nan
        else:
            mean_mse = np.nanmean([res['MSE'] for res in fold_results])
            mean_mae = np.nanmean([res['MAE'] for res in fold_results])
            mean_r2 = np.nanmean([res['R-kwadrat'] for res in fold_results])

        results_prophet_metrics.append({
            "Waluta": target_currency,
            "Lag BRENT": brent_lag_val,
            "Lags Waluty": LAGS_CURRENCY,
            "Mean MSE": mean_mse,
            "Mean MAE": mean_mae,
            "Mean R-kwadrat": mean_r2
        })

        # Wizualizacja prognoz
        predictions_to_plot = all_predictions[all_predictions.index >= pd.to_datetime(START_PLOT_DATE)].dropna()
        # Musimy pobrać rzeczywiste wartości z oryginalnego df_prophet_ready dla dopasowanych dat
        real_to_plot = df_prophet_ready.set_index('ds')['y'].loc[predictions_to_plot.index].dropna()

        # Ensure common index for plotting
        common_index = predictions_to_plot.index.intersection(real_to_plot.index)
        predictions_to_plot = predictions_to_plot.loc[common_index]
        real_to_plot = real_to_plot.loc[common_index]

        if not predictions_to_plot.empty and not real_to_plot.empty:
            plt.figure(figsize=(15, 7))
            plt.plot(real_to_plot.index, real_to_plot, label=f'Rzeczywiste {target_currency}')
            plt.plot(predictions_to_plot.index, predictions_to_plot, label=f'Prognozy Prophet (Lag={brent_lag_val})', color='orange')
            plt.title(f'Prognozy FP dla {target_currency} (Lag={brent_lag_val})')
            plt.xlabel('Data')
            plt.ylabel(f'Kurs {target_currency}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            filename = os.path.join(OUTPUT_CHARTS_PATH,
                                    f"prophet_forecast_brent_fx_lags_brentlag{brent_lag_val}_{target_currency.replace('/', '_')}_ts_cv.png")
            plt.savefig(filename)
            plt.close()

# Zapis wyników do Excela
results_df_prophet = pd.DataFrame(results_prophet_metrics)
results_df_prophet.to_excel(OUTPUT_RESULTS_EXCEL_PATH, index=False)

print(f"\nOcena modelu Prophet zakończona. Wyniki zapisano w: {OUTPUT_RESULTS_EXCEL_PATH}")
print(f"Wykresy zapisano w: {OUTPUT_CHARTS_PATH}")