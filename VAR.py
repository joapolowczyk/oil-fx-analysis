### VAR ####

# IMPACT OF OIL PRICES ON EXCHANGE RATES #

import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# Data load and preparation
df = pd.read_excel("Brent_fxrates.xlsx")
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df.asfreq("D").interpolate() #interpolation in order to fill blanks

# Selection of columns for VAR analysis and differentiation
df_var = df[["BRENT", "USD/EUR", "USD/CAD", "USD/NOK", "USD/RUB"]]
df_var_diff = df_var.diff().dropna()

# ADF TEST #
def check_stationarity(series):
    result = adfuller(series)
    return pd.Series({
        'Statystyka ADF': result[0],
        'Wartość p': result[1],
        'Krytyczna wartość 1%': result[4]['1%'],
        'Krytyczna wartość 5%': result[4]['5%'],
        'Krytyczna wartość 10%': result[4]['10%'],
        'Stacjonarna': result[1] <= 0.05
    })

# Checking stationarity before and after differentiation
adf_results = {
    'ADF Przed Różnicowaniem': pd.DataFrame({col: check_stationarity(df_var[col]) for col in df_var.columns}).T,
    'ADF Po Różnicowaniu': pd.DataFrame({col: check_stationarity(df_var_diff[col]) for col in df_var_diff.columns}).T
}

# Saving ADF test
output_path_adf = r"VAR\wyniki_adf.xlsx"
with pd.ExcelWriter(output_path_adf, mode='w') as writer:
    for sheet_name, df_result in adf_results.items():
        df_result.to_excel(writer, sheet_name=sheet_name)
print(f"\nWyniki testów ADF zostały zapisane do pliku: {output_path_adf}")

# Selection of the optimal order of delay for the VAR model
model = VAR(df_var_diff)
optimal_lag = model.select_order(maxlags=3).selected_orders['aic']

# Fitting the VAR model
model_fitted = model.fit(optimal_lag)

# Saving results to Excel
output_path_var_fit = r"VAR\wyniki_var.xlsx"
with pd.ExcelWriter(output_path_var_fit, mode='w') as writer:
    # Zapisz całe podsumowanie jako tekst
    summary_text = str(model_fitted.summary())
    pd.Series([summary_text]).to_excel(writer, sheet_name='Podsumowanie_Modelu', index=False, header=False)
    # Zapisz współczynniki
    model_fitted.params.to_excel(writer, sheet_name='Współczynniki_VAR')
print(f"VAR results have been saved to file: {output_path_var_fit}")

print("\nVAR analysis saved")


# FORECASTING #

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, mse
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib

matplotlib.use('Agg')

# Paths and parameters
DATA_DIR = "."
VAR_FINAL_DIR = os.path.join(DATA_DIR, "VAR")
VAR_CHARTS_DIR = os.path.join(VAR_FINAL_DIR, "charts")
os.makedirs(VAR_CHARTS_DIR, exist_ok=True) # Create a directory for charts

INPUT_FILE_PATH = os.path.join(DATA_DIR, "Brent_fxrates.xlsx")
OUTPUT_METRICS_EXCEL_PATH = os.path.join(VAR_FINAL_DIR, "var_predictions.xlsx")

TRAIN_END_DATE = '2017-12-31'
START_PLOT_DATE = '2018-01-01'
LAGS_TO_FORECAST = [1, 2, 3]

# Data load and preparation
df = pd.read_excel(INPUT_FILE_PATH)
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df.asfreq("D").interpolate()

currencies = ["USD/EUR", "USD/CAD", "USD/NOK", "USD/RUB"]
df_var_base = df[["BRENT"] + currencies].copy()

# Data split and differentiation
train_data_base = df_var_base[:TRAIN_END_DATE]
test_data_base = df_var_base[pd.to_datetime(TRAIN_END_DATE) + pd.Timedelta(days=1):]

train_data_diff = train_data_base.diff().dropna()

# Selection of the order of delay
model_full_data = VAR(df_var_base.diff().dropna())
lag_selection_report = model_full_data.select_order(maxlags=3)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {'RMSE': np.nan, 'MSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
    return {
        'RMSE': rmse(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

# List for metricks
all_results_metrics = []

writer = pd.ExcelWriter(OUTPUT_METRICS_EXCEL_PATH, engine='xlsxwriter')

# Prediction and evaluation loop for each lag
for current_lag in LAGS_TO_FORECAST:
    if len(train_data_diff) < current_lag:
        for col in df_var_base.columns:
            all_results_metrics.append({'Lag': current_lag, 'Zmienna': col,
                                        'RMSE_diff': np.nan, 'MSE_diff': np.nan, 'MAE_diff': np.nan, 'R2_diff': np.nan,
                                        'RMSE_level': np.nan, 'MSE_level': np.nan, 'MAE_level': np.nan, 'R2_level': np.nan})
        continue

    # VAR adjustment
    model_var = VAR(train_data_diff)
    model_fitted = model_var.fit(current_lag)

    # Forecasting differences
    last_observations_diff = train_data_diff.tail(current_lag).values
    steps_to_forecast = len(test_data_base)

    forecast_diff_array = model_fitted.forecast(y=last_observations_diff, steps=steps_to_forecast)
    forecast_diff_df = pd.DataFrame(forecast_diff_array, index=test_data_base.index, columns=df_var_base.columns)

    # Reversal of differentiation for forecasts
    forecast_level_df = pd.DataFrame(index=test_data_base.index, columns=df_var_base.columns)
    last_train_level = train_data_base.iloc[-1]

    # Implementation of reversal of differentiation
    for col in df_var_base.columns:
        forecast_level_df[col] = last_train_level[col] + forecast_diff_df[col].cumsum()


    # Evaluation and recording of metrics
    current_lag_metrics = []
    for column in df_var_base.columns:
        actual_diff_for_column = test_data_base[column].diff().dropna().reindex(forecast_diff_df.index).dropna()
        predicted_diff_for_column = forecast_diff_df[column].reindex(actual_diff_for_column.index).dropna()


        metrics_diff = calculate_metrics(actual_diff_for_column.values, predicted_diff_for_column.values)


        # Levels
        actual_level_for_column = test_data_base[column].reindex(forecast_level_df.index).dropna()
        predicted_level_for_column = forecast_level_df[column].reindex(actual_level_for_column.index).dropna()


        metrics_level = calculate_metrics(actual_level_for_column.values, predicted_level_for_column.values)

        result_row = {'Lag': current_lag, 'Zmienna': column}
        result_row.update({f'{k}_diff': v for k, v in metrics_diff.items()})
        result_row.update({f'{k}_level': v for k, v in metrics_level.items()})
        all_results_metrics.append(result_row)
        current_lag_metrics.append({'Zmienna': column, **metrics_diff, **metrics_level})

    pd.DataFrame(current_lag_metrics).to_excel(writer, sheet_name=f'VAR_Metrics_Lag_{current_lag}', index=False)

    # Generating charts
    plt.style.use('seaborn-v0_8-whitegrid')
    for column in df_var_base.columns:
        plt.figure(figsize=(16, 8))
        plt.plot(test_data_base.index, test_data_base[column], label=f'Rzeczywiste {column}')
        plt.plot(forecast_level_df.index, forecast_level_df[column], label=f'Prognozy VAR (Lag={current_lag})', color='orange')
        plt.xlim(pd.to_datetime(START_PLOT_DATE), forecast_level_df.index.max())
        plt.title(f'Prognozy VAR dla {column} (Lag={current_lag})')
        plt.xlabel('Data')
        plt.ylabel(f'Kurs {column}' if column != 'BRENT' else 'Cena Ropy Brent')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename_plot = os.path.join(VAR_CHARTS_DIR, f'forecast_var_levels_single_split_lag{current_lag}_{column.replace("/", "_")}.png')
        plt.savefig(filename_plot)
        plt.close()

# Data save
pd.DataFrame(all_results_metrics).to_excel(writer, sheet_name='Summary_All_Lags', index=False)
writer.close()

print(f"\nForecasting finished. Charts and results have been saved: {VAR_FINAL_DIR}")