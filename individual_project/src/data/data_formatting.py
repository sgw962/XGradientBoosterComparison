import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pytrends.exceptions import TooManyRequestsError
from pytrends.request import TrendReq
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import requests
import time
from abc import ABC, abstractmethod


class DataFormat(ABC):
    def __init__(self, df, df_name, keywords, timeframe, location):
        self.df = df
        self.df_name = df_name
        if len(keywords) != 3:
            raise ValueError('Only 3 keywords should be given')
        else:
            self.keywords = keywords
        self.timeframe = timeframe
        self.trends = None
        self.merged_df = None
        self.location = location

    @abstractmethod
    def create_target(self):
        pass

    def get_trends(self):
        self.keywords = [i.title() for i in self.keywords]
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
        new_trends = self.trends.reset_index()
        stock_data.loc[:, 'Exchange Date'] = pd.to_datetime(stock_data['Exchange Date'])
        new_trends.loc[:, 'date'] = pd.to_datetime(new_trends['date'])
        stock_data.loc[:, 'nearest_date'] = stock_data['Exchange Date'].apply(lambda x: new_trends['date'].iloc[(new_trends['date'] - x).abs().argsort()[0]])
        self.merged_df = pd.merge(stock_data, new_trends, left_on='nearest_date', right_on='date', how='left')
        self.merged_df.drop(['nearest_date', 'date', 'isPartial'], axis=1, inplace=True)

        columns = self.merged_df.columns.tolist()
        columns.remove(self.target)
        columns.append(self.target)
        self.merged_df = self.merged_df[columns]

        return self.merged_df

    @abstractmethod
    def normalise_data(self):
        pass

    def return_data(self):
        """
        Uses class functions to produce merged dataset
        """
        stocks = self.create_target()
        self.get_trends()
        self.merge_datasets(stocks)
        self.normalise_data()
        self.merged_df.to_excel(f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Individual Project/Individual_Project_Code/individual_project/data/Formatted Data/{self.df_name} {self.data_type} Data.xlsx', index=False)
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
        price = self.merged_df['Close']
        time_scale = self.merged_df['Exchange Date']
        plt.plot(time_scale, price)

        plt.title(f'{self.df_name} Normalised Daily Close Price Over Time')
        plt.xlabel('Exchange Date')
        plt.ylabel('Close Price')
        plt.show()

class ClassificationData(DataFormat):
    def __init__(self, df, df_name, keywords, timeframe, location):
        super().__init__(df, df_name, keywords, timeframe, location)
        self.target = 'Next Day Price Change'
        self.data_type = 'Classification'

    def create_target(self):
        self.df = self.df.dropna()
        self.df['Next Day Price Change'] = (self.df['%Chg'].shift(1) > 0).astype(int)
        self.df.drop(self.df.index[0], axis=0, inplace=True)
        return self.df

    def normalise_data(self):
        if self.merged_df is None:
            raise Exception('Previous methods must first be run to create the merged dataset')
        else:
            numeric_cols = self.merged_df.select_dtypes(include=np.number).columns.tolist()
            numeric_cols.remove('Next Day Price Change')
            numeric_cols.remove('Net')
            numeric_cols.remove('%Chg')
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(self.merged_df[numeric_cols])
            self.merged_df[numeric_cols] = df_scaled
            return self.merged_df


class RegressionData(DataFormat):
    def __init__(self, df, df_name, keywords, timeframe, location):
        super().__init__(df, df_name, keywords, timeframe, location)
        self.target = 'Next Day Close'
        self.data_type = 'Regression'

    def create_target(self):
        self.df = self.df.dropna()
        self.df['Next Day Close'] = self.df['Close'].shift(1)
        self.df = self.df.iloc[1:]
        return self.df

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
