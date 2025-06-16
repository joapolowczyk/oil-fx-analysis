### ARIMAX ####

#ANALIZA WPŁYWU CEN ROPY NA KURSY WALUT #

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Wczytanie i przygotowanie danych
df = pd.read_excel("Brent_fxrates.xlsx")
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df.asfreq("D").interpolate() #interpolacja w celu usunięciu brakujących wartości

# Dodanie kolumn z lagami Brent
for lag in [1, 2, 3]:
    df[f"BRENT_lag{lag}"] = df["BRENT"].shift(lag)

waluty = ["USD/EUR", "USD/CAD", "USD/NOK", "USD/RUB"]
results = []
d = 1  # Stałe d, różnicowanie pierwszego stopnia, sprawdzone przy badaniach stacjonarności ADF

# Testowanie modeli ARIMA dla każdej waluty i laga
for waluta in waluty:
    for lag in [1, 2, 3]:
        df_model = df[[waluta, f"BRENT_lag{lag}"]].dropna()
        y, X = df_model[waluta], df_model[[f"BRENT_lag{lag}"]]

        for p in range(4):
            for q in range(4):
                model_fit = ARIMA(y, exog=X, order=(p, d, q)).fit()
                results.append({
                    "Waluta": waluta,
                    "Lag": lag,
                    "ARIMA Order": (p, d, q),
                    "Coef BRENT_lag": model_fit.params.get(f"BRENT_lag{lag}"),
                    "P-value BRENT_lag": model_fit.pvalues.get(f"BRENT_lag{lag}"),
                    "AIC": model_fit.aic
                })

# Analiza i zapis wyników
results_df = pd.DataFrame(results)
results_df.to_excel(r"ARIMAX\arimax_test.xlsx", index=False)

best_models = results_df.loc[results_df.groupby(['Waluta', 'Lag'])['AIC'].idxmin()]
best_models.to_excel(r"ARIMAX\najlepsze_modele_aic.xlsx", index=False)

print("Analiza zakończona. Wyniki zostały zapisane do plików Excel.")

#PROGNOZA KURSÓW WALUT#

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import os

# Funkcja do oceny prognozy
def evaluate_forecast(y_true, y_predicted):
    return {
        "MSE": mean_squared_error(y_true, y_predicted),
        "RMSE": sqrt(mean_squared_error(y_true, y_predicted)),
        "MAE": mean_absolute_error(y_true, y_predicted),
        "R-kwadrat": r2_score(y_true, y_predicted)
    }

# Wczytanie i przygotowanie danych
df = pd.read_excel("Brent_fxrates.xlsx")
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df.asfreq("D").interpolate()

# Generowanie kolumn z opóźnieniami BRENT
for lag in [1, 2, 3]:
    df[f"BRENT_lag{lag}"] = df["BRENT"].shift(lag)

# Wczytanie najlepszych rzędów ARIMAX
best_arima_orders_df = pd.read_excel(r"ARIMAX\najlepsze_modele_aic.xlsx")

# Ustawienie stylu wykresów
plt.style.use('seaborn-v0_8-whitegrid')

results_prediction = []
output_charts_path_arimax = r"ARIMAX\charts" # Ścieżka do katalogu wykresów

# Pętla przez najlepsze modele i generowanie prognoz
for index, row in best_arima_orders_df.iterrows():
    waluta = row['Waluta']
    lag = int(row['Lag'])
    order = eval(row['ARIMA Order'])
    brent_lag_col = f"BRENT_lag{lag}"

    df_model = df[[waluta, brent_lag_col]].dropna()
    train_data = df_model[:'2017-12-31'] #dane treningowe
    test_data = df_model['2018-01-01':] #dane testowe

    y_train, X_train = train_data[waluta], train_data[[brent_lag_col]]
    y_test, X_test = test_data[waluta], test_data[[brent_lag_col]]

    # Modelowanie i predykcja
    model_fit = ARIMA(y_train, exog=X_train, order=order).fit()
    predictions = model_fit.predict(start=y_test.index[0], end=y_test.index[-1], exog=X_test)

    # Ocena prognozy
    metrics = evaluate_forecast(y_test, predictions)
    results_prediction.append({
        "Waluta": waluta,
        "Lag BRENT": lag,
        "ARIMA Order": order,
        **metrics
    })

    # Generowanie i zapis wykresu
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label=f'Rzeczywiste {waluta}')
    plt.plot(predictions.index, predictions, label=f'Prognozy ARIMAX (Lag={lag})', color='orange')
    plt.title(f'Prognozy ARIMAX dla {waluta} (Lag={lag})')
    plt.xlabel('Data')
    plt.ylabel(f'Kurs {waluta}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_charts_path_arimax, f'forecast_arimax_{waluta.replace("/", "_")}_brentlag{lag}.png')
    plt.savefig(filename)
    plt.close()

# Zapisanie wyników oceny do Excela
results_df_prediction = pd.DataFrame(results_prediction)
results_df_prediction.to_excel(r"ARIMAX\arimax_prediction_evaluation.xlsx", index=False)

print("Proces prognozowania zakończony. Wykresy i wyniki zostały zapisane do plików.")