import requests
from bs4 import BeautifulSoup
from finvizfinance.screener.overview import Overview
from finvizfinance.screener.financial import Financial
from finvizfinance.screener.valuation import Valuation
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
import re

def get_financial_scores(ticker):
    try:
        hdr = {'User-Agent': 'Mozilla/5.0'}
        guru_symbol = ticker.replace('-', '.')

        guru_req = requests.get("https://www.gurufocus.com/stock/" + guru_symbol, headers=hdr)
        if guru_req.status_code != 200 and guru_req.status_code != 403:
            return None

        guru_soup = BeautifulSoup(guru_req.content, 'html.parser')

        score_names = ['Piotroski F-Score', 'Altman Z-Score', 'Beneish M-Score']
        score_values = []

        for val in score_names:
            try:
                found = guru_soup.find('a', string=re.compile(val))
                if found is None:
                    print(f"Couldn't find {val} for {ticker}")
                    score_value = np.nan
                else:
                    score_value = found.find_next('td').text
                    if val == 'Piotroski F-Score':
                        score_value = score_value.split('/')[
                            0].strip()  # strip is used to remove any leading or trailing white spaces
                    score_value = float(score_value)
            except Exception as e:
                try:
                    print(f'Failed to convert {score_value} to float for {val}. Error: {e}')
                except UnboundLocalError:
                    print(f"Failed before 'score_value' was assigned. Error: {e}")
                score_value = np.nan
            score_values.append(score_value)

        return {
            'Piotroski F-Score': score_values[0],
            'Altman Z-Score': score_values[1],
            'Beneish M-Score': score_values[2]
        }

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")
        return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


foverview = Overview()
ffinancial = Financial()
fvaluation = Valuation()

filters_dict = {'Price/Free Cash Flow':'Under 40','InstitutionalOwnership':'Under 80%',
                'EPS growththis year': 'Over 10%', 'Market Cap.': '+Mid (over $2bln)'}
foverview.set_filter(filters_dict=filters_dict)
dataframe = foverview.screener_view()
print("\nOverview:")
print(dataframe.head())
ffinancial.set_filter(filters_dict=filters_dict)
financial_df = ffinancial.screener_view()
print("\nFinancial:")
print(financial_df.head())
fvaluation.set_filter(filters_dict=filters_dict)
valuation_df = fvaluation.screener_view()
print("\nValuation:")
print(valuation_df.head())

guru_ls = ['Piotroski F-Score', 'Altman Z-Score', 'Beneish M-Score']
dataframe = dataframe.assign(**{key: [None]*len(dataframe) for key in guru_ls})

session = requests.Session()
max_attempts = 3

# tickers = dataframe['Ticker\n\n'][:4]  # Get the first 5 tickers from the dataframe
tickers = dataframe['Ticker\n\n']

for ticker in tqdm(tickers, desc='Retrieving scores', unit='ticker'):
    for attempt in range(max_attempts):
        try:
            scores = get_financial_scores(ticker)
            if scores is None:
                break  # If scores is None, we simply go to the next ticker
            for key, value in scores.items():
                dataframe.loc[dataframe['Ticker\n\n'] == ticker, key] = value
            break  # If successful, break the retry loop and move to the next ticker
        except Exception as e:
            print(f"Attempt {attempt+1} of {max_attempts} failed for {ticker}. Exception: {e}")
            time.sleep(2)  # Wait for 2 seconds before the next attempt
    else:  # This else clause will run if the for loop is exhausted, i.e., all attempts failed.
        print(f"All attempts to fetch data for {ticker} have failed.")

# Filter based on guru scores
dataframe = dataframe[dataframe['Piotroski F-Score'] >= 6]
dataframe = dataframe[dataframe['Altman Z-Score'] >= 1.81]
dataframe = dataframe[dataframe['Beneish M-Score'] <= -1.78]

print("\nAfter filtering out bad f-score, z-score and m-score:")
print(dataframe.head())

# Filter out specific industry and country
dataframe = dataframe[~dataframe['Industry'].str.contains("bank", na=False)]
dataframe = dataframe[~dataframe['Country'].str.contains("China", na=False)]

print("\nAfter filtering out Industry and Country")
print(dataframe.head())

merged_df = pd.merge(dataframe, financial_df, on='Ticker\n\n')
merged_df = pd.merge(merged_df, valuation_df, on='Ticker\n\n')

# Convert the financial data to numeric values for ranking
for col in ['Profit M', 'EPS this Y', 'P/FCF']:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# Rank based on various metrics
merged_df['PM_rank'] = merged_df['Profit M'].rank(ascending=False)
merged_df['EPS_rank'] = merged_df['EPS this Y'].rank(ascending=False)
merged_df['PFCF_rank'] = merged_df['P/FCF'].rank(ascending=True)

# Summarize ranks and sort
merged_df['Total_rank'] = merged_df['PM_rank'] + merged_df['EPS_rank'] + merged_df['PFCF_rank']
merged_df = merged_df.sort_values('Total_rank')

print("\nAfter merging")
print(merged_df.head())

# Limit to top 20 stocks
# merged_df = merged_df.head(20)

# Write to text file
with open('sorted_filtered_tickers.txt', 'w') as f:
    for ticker in tqdm(merged_df['Ticker\n\n'], desc='Writing to file', unit='ticker'):
        f.write(f"{ticker}\n")
