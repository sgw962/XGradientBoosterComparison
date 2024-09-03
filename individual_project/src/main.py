import pandas as pd
from individual_project.src.data.data_formatting import ClassificationData, RegressionData
from individual_project.src.models.model_building import ClassificationModel, RegressionModel


def format_data(data_name, trends_words, time_scale, area, model_type):
    data = pd.read_excel(
        f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Individual Project/Individual_Project_Code/individual_project/data/Price History/{data_name} Price History.xlsx')
    if model_type == 'Regression':
        create_data = RegressionData(data, data_name, trends_words, time_scale, area)
    elif model_type == 'Classification':
        create_data = ClassificationData(data, data_name, trends_words, time_scale, area)
    else:
        raise ValueError('Model type must be either a Regression or Classification')

    updated_data = create_data.return_data()
    create_data.visualise_price()
    create_data.visualise_correlation()
    print(updated_data)

def run_model(data_name, booster_type, model_type):
    data = pd.read_excel(
        f'/Users/seanwhite/OneDrive - University of Greenwich/Documents/Individual Project/Individual_Project_Code/individual_project/data/Formatted Data/{data_name} {model_type} Data.xlsx')
    if model_type == 'Regression':
        create_model = RegressionModel(data, data_name, booster_type)
    elif model_type == 'Classification':
        create_model = ClassificationModel(data, data_name, booster_type)
    else:
        raise ValueError('Model type must be either a Regression or Classification')

    create_model.split_data(0.8, 0.1, 0.1)
    create_model.train_model()
    create_model.get_metrics_dataframe()


if __name__ == '__main__':
    stock_name = 'FTSE100'
    trends_keywords = ['covid', 'stock', 'policy']
    time_period = '2019-03-31 2024-03-27'
    location = 'GB'
    model = 'Regression'
    booster = 'gbtree'

    try:
        format_data(stock_name, trends_keywords, time_period, location, model)
    except Exception as e:
        print('Error running data formatting class:', e)

    try:
        run_model(stock_name, booster, model)
    except Exception as e:
        print('Error running model building class:', e)
