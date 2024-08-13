import ns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBRegressor, XGBClassifier
import seaborn as sns
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, df, df_name, booster, target):
        self.df = df
        self.df['Exchange Date'] = pd.to_datetime(self.df['Exchange Date'])
        self.df_sorted = self.df.sort_values(by='Exchange Date')
        self.df_name = df_name
        self.booster = booster
        self.target = target

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None

        self.model = None
        self.y_pred = None
        self.y_val_pred = None

        self.test_dates = None
        self.val_dates = None

        self.metrics = {}

    def split_data(self, train_size=0.8, test_size=0.1, val_size=0.1):
        assert train_size + test_size + val_size == 1, "Proportions must sum to 1"

        total_samples = len(self.df_sorted)
        train_end_idx = int(total_samples * train_size)
        test_end_idx = train_end_idx + int(total_samples * test_size)

        train_data = self.df_sorted.iloc[:train_end_idx]
        test_data = self.df_sorted.iloc[train_end_idx:test_end_idx]
        val_data = self.df_sorted.iloc[test_end_idx:]

        self.X_train = train_data.drop(['Exchange Date', self.target], axis=1)
        self.y_train = train_data.iloc[:, -1]
        self.X_test = test_data.drop(['Exchange Date', self.target], axis=1)
        self.y_test = test_data.iloc[:, -1]
        self.X_val = val_data.drop(['Exchange Date', self.target], axis=1)
        self.y_val = val_data.iloc[:, -1]

        self.test_dates = test_data['Exchange Date']
        self.val_dates = val_data['Exchange Date']

        print("Training data length:", len(train_data))
        print("Testing data length:", len(test_data))
        print("Validation data length:", len(val_data))

    @abstractmethod
    def build_model(self, parameters=None, training_x=None, training_y=None, testing_x=None):
        """Abstract method to be implemented in subclasses for building the model."""
        pass

    @abstractmethod
    def evaluate_model(self, y_true, y_pred, set_name="Test Set"):
        """Abstract method to be implemented in subclasses for evaluating the model."""
        pass

    @abstractmethod
    def train_model(self, params=None):
        """Abstract method to be implemented in subclasses for training the model."""
        pass

    def get_metrics_dataframe(self):
        metrics_table = pd.DataFrame(self.metrics).transpose()
        metrics_table.to_excel(f'Metrics for {self.df_name} using {self.booster} booster.xlsx')
        return metrics_table

    def feature_importance(self, features_name):
        if self.model is None:
            print('Cannot show feature importance as model has not been built yet. Please call build_model() first.')
            return
        else:
            feature_names = self.df_sorted.drop(['Exchange Date', self.target], axis=1).columns
            importances = self.model.feature_importances_

            feature_importances = dict(zip(feature_names, importances))
            print('\nFeature Importances:')
            for feature, importance in feature_importances.items():
                print(f'{feature}: {importance}')
            if np.any(importances < 0):
                print('Warning: Some feature importances are negative')

            sorted_indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(8, 6))
            plt.title(f'{features_name} Feature Importance', fontsize=16)
            plt.bar(range(len(importances)), importances[sorted_indices], align='center')
            plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=45)
            plt.tight_layout()
            plt.show()


class RegressionModel(BaseModel):
    def __init__(self, df, df_name, booster, target):
        super().__init__(df, df_name, booster, target)
        interaction = (df.iloc[:, 6] + df.iloc[:, 7] + df.iloc[:, 8]) / 3
        df.insert(len(df.columns) - 1, 'Trends', interaction)
        columns_to_drop = df.columns[6:9]
        df.drop(columns=columns_to_drop, inplace=True)

    def build_model(self, parameters=None, training_x=None, training_y=None, testing_x=None):
        if training_x is None:
            raise Exception("Please provide training data")

        if parameters is None:
            #parameters = {'updater': 'coord_descent', 'n_estimators': 1100, 'learning_rate': 0.3, 'lambda': 0.0001, 'feature_selector': 'greedy', 'booster': self.booster, 'alpha': 0, 'objective': 'reg:squarederror'}
            parameters = {'n_estimators': 1100, 'learning_rate': 0.3, 'lambda': 0.0001, 'booster': self.booster, 'alpha': 0}
        self.model = XGBRegressor(**parameters)
        self.model.fit(training_x, training_y)
        predicted_y = self.model.predict(testing_x)
        return predicted_y

    def evaluate_model(self, y_true, y_pred, set_name="Test Set"):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        self.metrics[set_name] = {'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

        print(f'{set_name} - \nMean Squared Error: {mse}\nRoot Mean Squared Error: {rmse}\nR^2 Score: {r2}\nMean Absolute Error: {mae}\nMean Absolute Percentage Error: {mape}%')

    def line_plot(self, data_name, time_frame, actual, predicted):
        plt.figure(figsize=(8, 6))
        plt.plot(time_frame, predicted, label='Predicted Values')
        plt.plot(time_frame, actual, label='Actual Values')

        plt.title(f'Actual vs Predicted Values {data_name}')
        plt.xlabel('Exchange Date')
        plt.ylabel('Next Day Close Price')
        plt.legend()
        plt.show()

    def scatter_plot(self, title, test_y, pred_y):
        plt.figure(figsize=(8, 6))
        plt.scatter(test_y, pred_y, alpha=0.5)
        plt.title(f'Actual vs Predicted {title}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=2)
        plt.show()

    def train_model(self, params=None):
        self.y_pred = self.build_model(params, self.X_train, self.y_train, self.X_test)
        self.evaluate_model(self.y_test, self.y_pred, '\nTest Set')

        self.y_val_pred = self.build_model(params, self.X_train, self.y_train, self.X_val)
        self.evaluate_model(self.y_val, self.y_val_pred, '\nValidation Set')

        self.line_plot(f'{self.df_name} Test Data', self.test_dates, self.y_test, self.y_pred)
        self.line_plot(f'{self.df_name} Validation Data', self.val_dates, self.y_val, self.y_val_pred)
        self.scatter_plot(f'{self.df_name} Test Data', self.y_test, self.y_pred)
        self.scatter_plot(f'{self.df_name} Validation Data', self.y_val, self.y_val_pred)

        self.feature_importance(f'{self.df_name} X Train Set')

    def retrain_with_validation(self, params=None):
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.y_pred = self.build_model(params, X_combined, y_combined, self.X_test)
        self.evaluate_model(self.y_test, self.y_pred, "\nTest Set After Retraining with Validation")

        self.line_plot(f'{self.df_name} Test Data After Retraining with Validation', self.test_dates, self.y_test, self.y_pred)
        self.scatter_plot(f'{self.df_name} Test Data After Retraining with Validation', self.y_test, self.y_pred)

        self.feature_importance(f'{self.df_name} X Train with Validation Set')

    def retrain_without_trends(self, params=None):
        columns_to_drop = [5]
        new_train = self.X_train.drop(self.X_train.columns[columns_to_drop], axis=1)
        new_test = self.X_test.drop(self.X_test.columns[columns_to_drop], axis=1)
        new_val = self.X_val.drop(self.X_val.columns[columns_to_drop], axis=1)

        self.y_pred = self.build_model(params, new_train, self.y_train, new_test)
        self.evaluate_model(self.y_test, self.y_pred, '\nTest Set Without Trends Data')

        self.y_val_pred = self.build_model(params, new_train, self.y_train, new_val)
        self.evaluate_model(self.y_val, self.y_val_pred, '\nValidation Set Without Trends Data')

        self.line_plot(f'{self.df_name} Test Set Without Trends', self.test_dates, self.y_test, self.y_pred)
        self.line_plot(f'{self.df_name} Validation Data Without Trends', self.val_dates, self.y_val, self.y_val_pred)
        self.scatter_plot(f'{self.df_name} Test Set Without Trends', self.y_test, self.y_pred)
        self.scatter_plot(f'{self.df_name} Validation Data Without Trends', self.y_val, self.y_val_pred)

        self.feature_importance(f'{self.df_name} X Train Set Without Trends')


class ClassificationModel(BaseModel):
    def __init__(self, df, df_name, booster, target):
        super().__init__(df, df_name, booster, target)
        interaction = (df.iloc[:, 10] + df.iloc[:, 11] + df.iloc[:, 12]) / 3
        df.insert(len(df.columns) - 1, 'Trends', interaction)
        columns_to_drop = df.columns[10:13]
        df.drop(columns=columns_to_drop, inplace=True)

    def compute_rsi(self, data, window):
        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def engineer_features(self):
        df = self.df_sorted
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['RSI_14'] = self.compute_rsi(df, 14)
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Trend'] = np.where(df['Close'] > df['Close'].shift(1), 1, 0)
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Change'] = df['Volume'].pct_change()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        self.df_sorted = df
        columns = self.df_sorted.columns.tolist()
        columns.remove('Next Day Price Change')
        columns.append('Next Day Price Change')
        self.df_sorted = self.df_sorted[columns]
        return self.df_sorted

    def build_model(self, parameters=None, training_x=None, training_y=None, testing_x=None):
        if training_x is None:
            raise Exception("Please provide training data")

        if parameters is None:
            parameters = {'verbosity': 2, 'seed': 42, 'scale_pos_weight': 1, 'nthread': 4, 'n_estimators': 500, 'learning_rate': 0.2, 'booster': self.booster}
        self.model = XGBClassifier(**parameters)
        self.model.fit(training_x, training_y)
        predicted_y = self.model.predict(testing_x)
        return predicted_y

    def evaluate_model(self, y_true, y_pred, set_name="Test Set"):
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

        self.metrics[set_name] = {'F1': f1, 'Recall': recall, 'Precision': precision}

        print(f'{set_name} - \nF1: {f1}\nRecall: {recall}\nPrecision: {precision}')

    def train_model(self, params=None):
        self.y_pred = self.build_model(params, self.X_train, self.y_train, self.X_test)
        self.evaluate_model(self.y_test, self.y_pred, '\nTest Set')

        self.y_val_pred = self.build_model(params, self.X_train, self.y_train, self.X_val)
        self.evaluate_model(self.y_val, self.y_val_pred, '\nValidation Set')

        self.plot_confusion_matrix(f'{self.df_name} Test Data', self.y_test, self.y_pred)
        self.plot_confusion_matrix(f'{self.df_name} Validation Data', self.y_val, self.y_val_pred)
        self.plot_roc_curve(f'{self.df_name} Test Data', self.X_test, self.y_test)
        self.plot_roc_curve(f'{self.df_name} Validation Data', self.X_val, self.y_val)
        self.plot_precision_recall(f'{self.df_name} Test Data', self.X_test, self.y_test)
        self.plot_precision_recall(f'{self.df_name} Validation Data', self.X_val, self.y_val)

        self.feature_importance(f'{self.df_name} X Train Set')

    def plot_confusion_matrix(self, data_name, actual, predicted):
        cm = confusion_matrix(actual, predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix for {data_name}')
        plt.show()

    def plot_precision_recall(self, data_name, X, y):
        y_scores = self.model.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, y_scores)
        plt.figure()
        plt.plot(recall, precision, marker='.')
        plt.title(f'Precision-Recall Curve for {data_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()

    def plot_roc_curve(self, data_name, X, y):
        y_scores = self.model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {data_name}')
        plt.legend(loc="lower right")
        plt.show()

#data = pd.read_excel('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Individual Project/Individual_Project_Code/individual_project/data/FTSE100 Classification Data.xlsx')
data = pd.read_excel('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/group_project_code/data/stocks & trends/Ocado Stock & trends.xlsx')
create_model = RegressionModel(data, 'Ocado', 'dart', 'Next Day Close')
#create_model.engineer_features()

create_model.split_data(0.8, 0.1, 0.1)

params = {'verbosity': [1],
          'seed': [42],
          'scale_pos_weight': [3, 4, 5, 6],
          'nthread': [4],
          'n_estimators': [200, 600, 900, 1000],
          'learning_rate': [0.075, 0.1, 0.2],
          'booster': ['gbtree', 'gblinear', 'dart'],
          'alpha': [0.001, 0.01, 0.1],
          'lambda': [0.001, 0.01, 0.1]
          }

#best_params = create_model.tune_parameters(param_grid)
create_model.train_model()#best_params)
create_model.get_metrics_dataframe()
