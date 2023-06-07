import pandas as pd
import pytz
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler


# define function to load dataset and do some preprocessing
def load_file(file):
    file_path = file
    df = pd.read_excel(file_path)
    return(df)

def localize_tz(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df.index = pd.bdate_range(start=df.index[0], periods=len(df)) # This will define the frequency of data on the basis of weekdays
    df = df.set_index(df.Date) # set date as index
    df = df.tz_localize(pytz.timezone('Europe/Lisbon'))
    df.drop(columns = "Date", inplace = True)
    return df

def additive_decom(df):
    rcParams['figure.figsize'] = 18, 8  
    series = df["Close"]
    result = seasonal_decompose(series, model='additive', period=int(len(series)/series.index.year.nunique()), extrapolate_trend= "freq",) # if i have to see the yearly seasonal pattern i need to divide total data point by number of year. 
    df["Trend"] = result.trend
    df["Seasonality"] = result.seasonal
    df["Residual"] = result.resid
    # Splitting data in to target, features and taking seasonality seperately, Looking at the pattern of stock seasonality pattern is constant every year so we will take it now.
    df["target"] = df["Close"].shift(-1).fillna(0)
    return df 

def calculate_technical_indicators(df):
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window=14).mean()
    avg_loss = down.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # EMA
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema24 = df['Close'].ewm(span=24, adjust=False).mean()
    
    # MACD
    macd_line = ema12 - ema24
    
    # Moving Averages
    ma12 = df['Close'].rolling(window=12).mean()
    ma24 = df['Close'].rolling(window=24).mean()
    
    # BIAS
    bias12 = (df['Close'] - ma12) / ma12 * 100
    bias24 = (df['Close'] - ma24) / ma24 * 100
    
    # Create a new DataFrame with the calculated indicators
    indicators = pd.DataFrame({
        'RSI': rsi,
        'EMA12': ema12,
        'EMA24': ema24,
        'MACD Line': macd_line,
        'MA12': ma12,
        'MA24': ma24,
        'BIAS12': bias12,
        'BIAS24': bias24
    }, index=df.index)
    
    
    indicator = indicators.fillna(0)
    df = df.merge(indicator, left_index= True, right_index= True)
    
    return df


# Separate features and target 
def split_data(df):
    X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'RSI', 'EMA12',
        'EMA24', 'MACD Line', 'MA12', 'MA24',
        'BIAS12', 'BIAS24', 'Trend',
        'Seasonality', 'Residual']]
    y = df["target"]
    return X, y

def fit_scaler(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
    
    inverse_scaler = MinMaxScaler()
    inverse_scaler.fit(y_scaled)

    return X_scaled, y_scaled, inverse_scaler

