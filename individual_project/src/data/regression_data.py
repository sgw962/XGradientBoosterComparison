import ns
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import seaborn as sns
import numpy as np
from pytrends.exceptions import TooManyRequestsError
from pytrends.request import TrendReq
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import requests
import time


class CreateRegressionData:
    """
    This class takes as input an Eikon historical stock price dataset as well as 3 keywords and a and timeframe (which
    shouldn't go beyond 5 years if wanting weekly data) with which to create the Google Trends data. It will return the
    aggregated dataset where the weekly trends values are paired with the nearest daily stock price value.
    The methods that should be called are:
    - return_data() which will create the merged dataframe
    - visualise_correlation() will produce a correlation matrix with the target and features columns (removing date).
    - visualise_price() will produce a line graph of the closing price column over the five-year period.
    """
    def __init__(self, df, df_name, keywords, timeframe, location):
        self.df = df
        self.df_name = df_name
        # Ensures there are 3 keywords entered
        if len(keywords) != 3:
            raise ValueError('Only 3 keywords should be given')
        else:
            self.keywords = keywords
        self.timeframe = timeframe
        self.trends = None
        self.merged_df = None
        self.location = location

    def create_target(self):
        self.df = self.df.dropna()
        # Creates target column
        self.df['Next Day Close'] = self.df['Close'].shift(1)
        self.df = self.df.iloc[1:]
        return self.df

    def get_trends(self):
        self.keywords = [i.title() for i in self.keywords]
        # Will need to be adjusted for use on your own computer
        vader_lexicon_path = '/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/group_project_code/vader_lexicon.txt'
        sia = SentimentIntensityAnalyzer(
            lexicon_file=vader_lexicon_path
        )

        pytrends = TrendReq(hl='en-Uk', tz=60)

        retries = 5
        delay = 10

        for attempt in range(retries):
            try:
                pytrends.build_payload(self.keywords, timeframe=self.timeframe, geo=self.location)
                self.trends = pytrends.interest_over_time()
                return pd.DataFrame(self.trends)
            except TooManyRequestsError as e:
                print(f"Attempt {attempt + 1}/{retries}: Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            except requests.exceptions.RequestException as e:
                print(f"Request exception: {e}")
                raise e
        raise Exception("Failed to retrieve trends data after multiple attempts")

    def merge_datasets(self, stock_data):
        # Resetting index for trends data so the date column is no longer used as the index and can instead be used to merge the datasets
        new_trends = self.trends.reset_index()

        # Ensuring both date columns are in datetime format
        stock_data['Exchange Date'] = pd.to_datetime(stock_data['Exchange Date'])
        new_trends['date'] = pd.to_datetime(new_trends['date'])
        # Finds the nearest 'Exchange Date' value to apply to trends date
        stock_data['nearest_date'] = stock_data['Exchange Date'].apply(lambda x: new_trends['date'].iloc[(new_trends['date'] - x).abs().argsort()[0]])
        # Joins two datasets together
        self.merged_df = pd.merge(stock_data, new_trends, left_on='nearest_date', right_on='date', how='left')

        # Removing redundant columns
        self.merged_df.drop(['nearest_date', 'date', 'isPartial', 'Net', '%Chg', 'Volume', 'Turnover - GBP'], axis=1, inplace=True)

        # Adjusting the columns order to have 'Next Day Close' on the end
        columns = self.merged_df.columns.tolist()
        columns.remove('Next Day Close')
        columns.append('Next Day Close')
        self.merged_df = self.merged_df[columns]

        return self.merged_df

    def normalise_data(self):
        if self.merged_df is None:
            raise Exception('Previous methods must first be run to create the merged dataset')
        else:
            numeric_cols = self.merged_df.select_dtypes(include=np.number).columns.tolist()
            numeric_cols.remove('Next Day Close')
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(self.merged_df[numeric_cols])
            self.merged_df[numeric_cols] = df_scaled
            return self.merged_df

    def return_data(self):
        """
        Uses classes functions to produce merged dataset
        """
        stocks = self.create_target()
        self.get_trends()
        self.merge_datasets(stocks)
        self.normalise_data()
        self.merged_df.to_excel(f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Individual Project/Individual_Project_Code/individual_project/data/{self.df_name} Regression Data.xlsx', index=False)
        return self.merged_df

    def visualise_correlation(self):
        """
        Produces correlation matrix
        """
        corr_df = self.merged_df.drop('Exchange Date', axis=1)
        corr = corr_df.corr()

        plt.figure(figsize=(10, 10))
        plt.title(f'{self.df_name} Data Correlation Matrix', fontsize=25)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.xticks(rotation=45)
        plt.show()

    def visualise_price(self):
        """
        Visualises target variable
        """
        price = self.merged_df['Next Day Close']
        time_scale = self.merged_df['Exchange Date']
        plt.plot(time_scale, price)

        plt.title(f'{self.df_name} Next Day Closing Price Over Time')
        plt.xlabel('Exchange Date')
        plt.ylabel('Next Day Close Price')
        plt.show()

data = pd.read_excel(f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Individual Project/Individual_Project_Code/individual_project/data/raw/FTSE100 Price History.xlsx')

create_data = CreateRegressionData(data, 'FTSE100', ['covid', 'stock', 'policy'], '2019-03-31 2024-03-27', 'GB')
updated_df = create_data.return_data()
create_data.visualise_correlation()
create_data.visualise_price()
