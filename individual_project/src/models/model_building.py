import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
import seaborn as sns
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, df, df_name, booster):
        self.df = df
        interaction = (self.df.iloc[:, 10] + self.df.iloc[:, 11] + self.df.iloc[:, 12]) / 3
        self.df.insert(len(self.df.columns) - 1, 'Trends', interaction)
        columns_to_drop = self.df.columns[10:13]
        self.df.drop(columns=columns_to_drop, inplace=True)

        self.df['Exchange Date'] = pd.to_datetime(self.df['Exchange Date'])
        self.df = self.df.sort_values(by='Exchange Date')
        self.df_name = df_name
        self.booster = booster

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

        total_samples = len(self.df)
        train_end_idx = int(total_samples * train_size)
        test_end_idx = train_end_idx + int(total_samples * test_size)

        train_data = self.df.iloc[:train_end_idx]
        test_data = self.df.iloc[train_end_idx:test_end_idx]
        val_data = self.df.iloc[test_end_idx:]

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
        directory = f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Individual Project/Individual_Project_Code/individual_project/data/Metrics/{self.df_name} Metrics/'
        os.makedirs(directory, exist_ok=True)
        file_path = f'{directory}{self.model_type} with {self.booster} on {self.df_name} data.xlsx'
        metrics_table.to_excel(file_path)
        return metrics_table

    def feature_importance(self):
        if self.model is None:
            print('Cannot show feature importance as model has not been built yet. Please call build_model() first.')
            return
        else:
            feature_names = self.df.drop(['Exchange Date', self.target], axis=1).columns
            importances = self.model.feature_importances_

            sorted_indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(8, 6))
            plt.title(f'{self.df_name} train data Feature Importance with {self.booster} {self.model_type}', fontsize=16)
            plt.bar(range(len(importances)), importances[sorted_indices], align='center')
            plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=45)
            plt.tight_layout()
            plt.show()

class RegressionModel(BaseModel):
    def __init__(self, df, df_name, booster):
        super().__init__(df, df_name, booster)
        self.target = 'Next Day Close'
        self.model_type = 'Regressor'

    def build_model(self, parameters=None, training_x=None, training_y=None, testing_x=None):
        if training_x is None:
            raise Exception("Please provide training data")

        if parameters is None:
            #parameters = {'n_estimators': 1000, 'learning_rate': 0.3, 'lambda': 0.001, 'alpha': 0.001, 'booster': self.booster}
            parameters = {'n_estimators': 1000, 'learning_rate': 0.3, 'lambda': 0.001, 'alpha': 0.001, 'booster': self.booster}
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

        self.metrics[set_name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

        print(f'{set_name} - \nRoot Mean Squared Error: {rmse}\nR^2 Score: {r2}\nMean Absolute Error: {mae}\nMean Absolute Percentage Error: {mape}')

    def line_plot(self, data_name, time_frame, actual, predicted):
        plt.figure(figsize=(8, 6))
        plt.plot(time_frame, predicted, label='Predicted Values')
        plt.plot(time_frame, actual, label='Actual Values')

        plt.title(f'Actual vs Predicted Values {data_name} using {self.booster}')
        plt.xlabel('Exchange Date')
        plt.ylabel('Next Day Close Price')
        plt.legend()
        plt.show()

    def scatter_plot(self, title, test_y, pred_y):
        plt.figure(figsize=(8, 6))
        plt.scatter(test_y, pred_y, alpha=0.5)
        plt.title(f'Actual vs Predicted {title} using {self.booster}')
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

        self.feature_importance()


class ClassificationModel(BaseModel):
    def __init__(self, df, df_name, booster):
        super().__init__(df, df_name, booster)
        self.target = 'Next Day Price Change'
        self.model_type = 'Classifier'

    def build_model(self, parameters=None, training_x=None, training_y=None, testing_x=None):
        if training_x is None:
            raise Exception("Please provide training data")

        if parameters is None:
            parameters = {'scale_pos_weight': 1.5, 'n_estimators': 800, 'learning_rate': 0.2, 'alpha': 0.001, 'lambda': 0.001, 'booster': self.booster}
        self.model = XGBClassifier(**parameters)
        self.model.fit(training_x, training_y)
        predicted_y = self.model.predict(testing_x)
        return predicted_y

    def evaluate_model(self, y_true, y_pred, set_name="Test Set"):
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        self.metrics[set_name] = {'F1': f1, 'Recall': recall, 'Precision': precision, 'Accuracy': accuracy}

        print(f'{set_name} - \nF1: {f1}\nRecall: {recall}\nPrecision: {precision}\nAccuracy: {accuracy}')

    def plot_confusion_matrix(self, data_name, actual, predicted):
        cm = confusion_matrix(actual, predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix for {data_name} using {self.booster}')
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
        plt.title(f'ROC Curve for {data_name} using {self.booster}')
        plt.legend(loc="lower right")
        plt.show()

    def train_model(self, params=None):
        self.y_pred = self.build_model(params, self.X_train, self.y_train, self.X_test)
        self.evaluate_model(self.y_test, self.y_pred, '\nTest Set')

        self.y_val_pred = self.build_model(params, self.X_train, self.y_train, self.X_val)
        self.evaluate_model(self.y_val, self.y_val_pred, '\nValidation Set')

        self.plot_confusion_matrix(f'{self.df_name} Test Data', self.y_test, self.y_pred)
        self.plot_confusion_matrix(f'{self.df_name} Validation Data', self.y_val, self.y_val_pred)
        self.plot_roc_curve(f'{self.df_name} Test Data', self.X_test, self.y_test)
        self.plot_roc_curve(f'{self.df_name} Validation Data', self.X_val, self.y_val)

        self.feature_importance()
