
### BRENT DOWNLOAD ###

import pandas as pd
import urllib.request
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

if not API_KEY:
    raise ValueError("API key not found. Make sure you have file .env with API_KEY")

#URL with Brent parameters
BASE_URL = (
    "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    "?frequency=daily"
    "&data[0]=value"
    "&facets[product][]=EPCBRENT"  #Product EPCBRENT (Brent)
    "&start=2010-01-01"  #Data start
    "&end=2019-12-31"    #Data end
    "&sort[0][column]=period"
    "&sort[0][direction]=desc"
)

all_data = []
offset = 0
limit = 5000

while True:

    url = f"{BASE_URL}&offset={offset}&length={limit}&api_key={API_KEY}"
    print(f"Pobieranie URL: {url}")
    with urllib.request.urlopen(url) as response:
        result = json.loads(response.read().decode())


    records = result["response"]["data"]
    if not records:
        break  #If empty end

    all_data.extend(records)
    offset += limit
    print(f"Downloaded: {len(all_data)} records...")

#Conversion to DataFrame
OIL = pd.DataFrame(all_data)
OIL["period"] = pd.to_datetime(OIL["period"])
OIL = OIL.sort_values("period")

print(OIL)

### FX RATES DOWNLOAD ###

import yfinance as yf

tickers_list = ['USDCAD=X', 'USDEUR=X', 'USDNOK=X', 'USDRUB=X']
all_data = yf.download(tickers_list, start='2010-01-01', end='2019-12-31')

cad = all_data.xs('USDCAD=X', level=1, axis=1).reset_index()
eur = all_data.xs('USDEUR=X', level=1, axis=1).reset_index()
nok = all_data.xs('USDNOK=X', level=1, axis=1).reset_index()
rub = all_data.xs('USDRUB=X', level=1, axis=1).reset_index()

### DATA CHANGE ###

eur = eur[['Date', 'Close']].rename(columns={'Close':'USD/EUR'})
cad = cad[['Date', 'Close']].rename(columns={'Close':'USD/CAD'})
nok = nok[['Date', 'Close']].rename(columns={'Close':'USD/NOK'})
rub = rub[['Date', 'Close']].rename(columns={'Close':'USD/RUB'})

eur['Date'] = pd.to_datetime(eur['Date'])
cad['Date'] = pd.to_datetime(cad['Date'])
nok['Date'] = pd.to_datetime(nok['Date'])
rub['Date'] = pd.to_datetime(rub['Date'])

merged = eur.merge(cad, on='Date').merge(nok, on='Date').merge(rub, on='Date')

########################################

df = OIL
df['Date'] = pd.to_datetime(df['period'])

df_grouped = df[['Date', 'value']].rename(columns={'value': 'BRENT'})

### Data merge ###

brent_prices = df_grouped
fxrates = merged

both = brent_prices.merge(fxrates, on='Date')

#Coversion to datetime type
both['Date'] = pd.to_datetime(both['Date'])

#Conversion to float type
for col in ['BRENT', 'USD/EUR', 'USD/CAD', 'USD/NOK', 'USD/RUB']:
    both[col] = pd.to_numeric(both[col], errors='coerce')

both[['USD/EUR', 'USD/CAD', 'USD/NOK', 'USD/RUB']] = both[['USD/EUR', 'USD/CAD', 'USD/NOK', 'USD/RUB']].round(4)

#Excel save
both.to_excel("Brent_fxrates.xlsx", index=False)
