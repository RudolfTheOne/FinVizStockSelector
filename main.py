import requests
from bs4 import BeautifulSoup
from finvizfinance.screener.overview import Overview
from finvizfinance.screener.financial import Financial
from finvizfinance.screener.valuation import Valuation
import pandas as pd
from tqdm import tqdm
import numpy as np

def get_financial_scores(ticker):
    try:
        url = f"https://www.gurufocus.com/stock/{ticker}/summary"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to fetch the web page: {url}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        piotroski_f_score_tag = soup.find(string='Piotroski F-Score')
        piotroski_f_score = piotroski_f_score_tag.find_next().text if piotroski_f_score_tag else np.nan

        altman_z_score_tag = soup.find(string='Altman Z-Score')
        altman_z_score = altman_z_score_tag.find_next().text if altman_z_score_tag else np.nan

        beneish_m_score_tag = soup.find(string='Beneish M-Score')
        beneish_m_score = beneish_m_score_tag.find_next().text if beneish_m_score_tag else np.nan

        return {
            'Piotroski F-Score': float(piotroski_f_score) if piotroski_f_score else np.nan,
            'Altman Z-Score': float(altman_z_score) if altman_z_score else np.nan,
            'Beneish M-Score': float(beneish_m_score) if beneish_m_score else np.nan
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

for ticker in tqdm(dataframe['Ticker\n\n'], desc='Retrieving scores', unit='ticker'):
    scores = get_financial_scores(ticker)
    for key, value in scores.items():
        dataframe.loc[dataframe['Ticker\n\n'] == ticker, key] = value


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
