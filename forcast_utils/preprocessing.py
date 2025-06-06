import os
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import jax.numpy as jnp
import numpy as np


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical data for a given ticker."""
    # check if the data is already downloaded
    if f"{ticker}_data.csv" in os.listdir():
        print(f"Data for {ticker} already downloaded.")
        dataframe = pd.read_csv(f"{ticker}_data.csv", index_col=0, parse_dates=True, date_format="%Y-%m-%d")
        # remove first two rows as they are not needed
        dataframe = dataframe.iloc[2:]
        # interpret all but the first column as float
        for col in dataframe.columns[1:]:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='raise')
        dataframe['Close'] = pd.to_numeric(dataframe['Close'], errors='raise')
        return dataframe


    data = yf.download(ticker, start=start, end=end)
    # save the data to a CSV file
    data.to_csv(f"{ticker}_data.csv")

    return data


def preprocess_data(data: pd.DataFrame, target_column: str, feature_columns: list, window: int = 30,) -> pd.DataFrame:
    """ This function will preprocess the data so that it is ready for training. """
    assert window > 0, "Window must be greater than 0."
    assert data is not None, "Data must not be None."

    data_temp = calculate_log_returns(data)
    data_temp = calculate_volatility(data_temp, base_column='Log_Returns', window=window)
    close_column = 'Close'
    base_column = f'Log_Returns_{window}_Volatility'

    data_temp = calculate_returns(data_temp, base_column=close_column)
    data_temp = calculate_volatility(data_temp, base_column=close_column + '_Returns', window=window)
    data_temp = calculate_window_variance(data_temp, volatility_column=base_column)

    variance_column = f'{base_column}_Variance'

    data_temp = add_target_column(data_temp, target_column=target_column, base_column=variance_column)
    data_temp = scale_data(data_temp, columns=feature_columns + [target_column] + [base_column] + [variance_column])

    data_temp = data_temp.dropna()
    data_temp = data_temp.reset_index(drop=True)
    data_temp = data_temp[feature_columns + [target_column, base_column, 'Log_Returns', f'{base_column}_Variance', 'Close_Returns', f'Close_Returns_{window}_Volatility']]
    return data_temp

def calculate_returns(data: pd.DataFrame, base_column: str) -> pd.DataFrame:
    """ This function will calculate the returns based on the base column. """
    assert base_column in data.columns, f"Data must contain '{base_column}' column"

    data[f'{base_column}_Returns'] = data[base_column].pct_change()
    return data

def calculate_window_variance(data: pd.DataFrame, volatility_column: str) -> pd.DataFrame:
    """ This function will calculate the rolling variance of the volatility column. """
    assert volatility_column in data.columns, f"Data must contain '{volatility_column}' column"

    data[f'{volatility_column}_Variance'] = np.square(data[volatility_column])
    return data

def add_target_column(data: pd.DataFrame, target_column: str, base_column:str) -> pd.DataFrame:
    """ This function will preprocess the data by adding a target column based on the base column. """
    assert base_column in data.columns, f"Data must contain '{base_column}' column"
    assert target_column not in data.columns, f"Data must contain '{target_column}' column"
    print("Using base column:", base_column, "to create target column:", target_column)
    # Add target column
    data[target_column] = data[base_column].shift(1)
    # Drop rows with NaN values
    data.dropna(inplace=True)

    return data

def scale_data(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """ This function will scale the data using MinMaxScaler. """

    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def calculate_volatility(data: pd.DataFrame, base_column: str,  window: int = 30) -> pd.DataFrame:
    """ This function will add a column with the volatitlity variance of the base Column."""
    assert base_column in data.columns, f"Data must contain '{base_column}' column"

    data[f'{base_column}_{window}_Volatility'] = data[base_column].rolling(window).std()
    return data

def calculate_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    """ This function will add a column with the log returns based on the 'Close' column. """
    assert 'Close' in data.columns, "Data must contain 'Close' column"
    data['temp'] = data['Close'].shift(1)
    data.dropna()
    data['Log_Returns'] = np.log(data['Close'] / data['temp'])
    data.drop(columns=['temp'], inplace=True)
    return data

def drop_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """ This function will drop the specified columns from the data. """
    assert isinstance(columns, list), "Columns must be a list"
    data = data.drop(columns=columns, errors='ignore')
    return data

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    start = "2020-01-01"
    end = "2023-01-01"
    data = download_data(ticker, start, end)
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Log_Returns_Volatility']
    target_column = 'Next_Day_Close'
    window = 30

    processed_data = preprocess_data(data, target_column, feature_columns, window)
    print(processed_data.head())