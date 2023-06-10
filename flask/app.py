from flask import Flask, render_template, request, session, send_file, Response, make_response
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import json
import datetime
import os
from io import StringIO
import csv
from utils import load_file, localize_tz, additive_decom, calculate_technical_indicators
from key import api_key
from datetime import datetime, timedelta
from keras.models import load_model
from sklearn.metrics import r2_score
import pytz
from pylab import rcParams
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import load_model
from werkzeug.utils import secure_filename

import joblib


app = Flask(__name__)
app.secret_key = 'Todays_Scession'  # Set the secret key for session management


# define route for home page
@app.route('/')
def home():
    logo_filename = "my_logo.png"
    ironhack_logo = "ironhack_logo.png"
    machine_learning = "machine_learning.png"
    action_plan = "action_plan.png"
    time_series = "time_series.png"
    outcome = "outcome.png"
    future = "future_improvement.png"
    namaste = "namaste.png"
    return render_template('index.html', logo_filename = logo_filename, ironhack_logo = ironhack_logo, machine_leraning  = machine_learning, action_plan= action_plan, time_series = time_series, outcome = outcome, future = future, namaste = namaste)


# Define route for prediction page
@app.route('/prediction.html', methods=['GET', 'POST'])
def predict():
     # Check if a file is uploaded
    logo_filename = "my_logo.png"
    if request.method == 'POST' and 'file' in request.files:
        # Get the uploaded file
        file = request.files['file']
        # Save the file to a specific location
        file_path = file.filename
        file.save(file_path)

        # Load the dataset and perform preprocessing
        df = load_file(file_path)
        df = localize_tz(df)
        df = additive_decom(df)
        df = calculate_technical_indicators(df)
        # import scaler used the scale during the training process
        
        # Prepare the features for prediction
        X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'RSI', 'EMA12',
        'EMA24', 'MACD Line', 'MA12', 'MA24',
        'BIAS12', 'BIAS24', 'Trend',
        'Seasonality', 'Residual']]
        y = df["target"]

        # Scale the features and target
        #scaler = MinMaxScaler()
        scaler_X = joblib.load("C:/Users/udaya/Desktop/Bootcamp/Project/X_scaler.save")  
        X_scaled = scaler_X.transform(X)
        # scaler_y = MinMaxScaler()
        # y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
        scaler_y = joblib.load("C:/Users/udaya/Desktop/Bootcamp/Project/y_scaler.save")
        y_scales = scaler_y.transform(y.values.reshape(-1, 1))

        

        # Prepare the input sequences
        window_size = 20
        X_test = []
        for i in range(window_size, len(X_scaled)):
            X_test.append(X_scaled[i - window_size:i])

        X_test = np.array(X_test)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        
        # Construct the absolute file path to load and save model
        base_path = os.path.abspath('C:/Users/udaya/Desktop/Bootcamp/Project/flask/galp_final.h5')
        loaded_model = load_model(base_path)
        # Make predictions
        pred = loaded_model.predict(X_test)                                 
        predicted_prices = pd.DataFrame(scaler_y.inverse_transform(pred))
        """predicted_price = predicted_prices.set_index(X[window_size:].index)
        y = pd.concat([y, predicted_price], axis=1).reindex(y.index)"""

        # Get the last predicted price
        last_predicted_price = predicted_prices.iloc[-1][0].round(2)

        # Claculate error and accuracy
        error = np.sqrt(mean_squared_error(y[window_size:], predicted_prices)).round(2)
        r2 = r2_score(y[window_size:], predicted_prices.values).round(2)
   
        # Create traces
        fig = go.Figure()
        
        # Create a Plotly trace for actual prices
        fig.add_trace(go.Scatter(x=X.iloc[window_size:-1].index,
                                y=y.iloc[window_size:-1],
                                mode='lines',
                                name='Closing Price'))
        
        # Create a Plotly trace for predicted prices
        fig.add_trace(go.Scatter(x=X.iloc[window_size:].index,
                                        y=predicted_prices[0],
                                    mode='lines',
                                name='Predicted Price'))
        # Set the layout with a date slider
        fig.update_layout(xaxis_rangeslider_visible=True)
        # Create the figure
        # figure = go.Figure(data=[actual_trace, predicted_trace], layout=layout)
        # Convert the figure to JSON and pass it to the template
        chart = fig.to_html(full_html=True)

        # Render the template with the prediction results
        return render_template('prediction.html', predict=True, last_predicted_price=last_predicted_price, logo_filename = logo_filename, chart=chart, r2 = r2, error = error)

    # Render the template without prediction results
    return render_template('prediction.html', predict=False, logo_filename = logo_filename)


# Define route for live-data page
@app.route('/live-data.html', methods=['GET', 'POST'])

def live_data():
    logo_filename = "my_logo.png"
    # Define the stock symbols you want to fetch data for
    symbols = ['ALTR.LS', 'BCP.LS', 'SLBEN.LS', 'CFN.LS', 'COR.LS', 'CTT.LS', 'EDP.LS', 'EDPR.LS', 'ESON.LS', 'FCP.LS', 'GALP.LS', 'GLINT.LS',
               'GVOLT.LS', 'IBS.LS', 'IPR.LS', 'INA.LS', 'JMT.LS', 'LIG.LS', 'MAR.LS', 'MCP.LS', 'MRL.LS', 'EGL.LS', 'NOS.LS', 'NBA.LS', 'PHR.LS',
               'RAM.LS', 'RED.LS', 'RENE.LS', 'SEM.LS', 'SON.LS', 'SNC.LS', 'SCP.LS', 'TDSA.LS', 'NVG.LS', 'VAF.LS', '^STOXX50E', '^EVZ', 'PSI20.LS']

    intervals = ["1d", "1h", "1m", "1wk", "1mo"]  # Available intervals

    if request.method == "POST":
        selected_stock = request.form["stockSelect"]
        selected_interval = request.form['intervalSelect']
    else:
        selected_stock = symbols[10]
        selected_interval = intervals[0]

    today = datetime.now().date()
    
    # Set the appropriate period and start date based on the selected interval
    if selected_interval == '1d':  # Daily interval
        period = '2y'
        start = today - timedelta(days=2*365-1)
    elif selected_interval == '1h':  # Hourly interval
        period = '30d'
        start = today - timedelta(days=30)
    elif selected_interval == '1m':  # Minute interval
        period = '1d'
        start = today - timedelta(days=5)
    elif selected_interval == '1wk':  # Weekly interval
        period = 'max'
        start = today - timedelta(days=400)
    else:   # Monthly interval
        period = 'max'
        start = None

    # Fetch live stock data using yfinance

    stock = yf.Ticker(selected_stock)

    # Check if the data is already stored in the session
    if 'stock_data' in session and session['stock_data'].get(selected_stock):
        data_json = session['stock_data'][selected_stock]
        data = pd.read_json(data_json, orient='index')
    else:
        data = stock.history(interval = selected_interval, period=period, start=start, end=today)

        # Store the fetched data in the session
        if 'stock_data' not in session:
            session['stock_data'] = {}
        data_json = data.to_json(orient='index')
        session['stock_data'][selected_stock] = data_json

    # Create a Plotly figure
    fig = go.Figure()

    # Add candlestick trace for price
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Price'))
    # Set the y-axis tick size for price
    tick_size_price = (data['High'].max() - data['Low'].min()) / 10
    # Set layout for the figure
    fig.update_layout(
        title_text='Stock Price and Volume',
        xaxis=dict(rangeslider=dict(visible=True), type='date'),
        yaxis=dict(title="Price", tickformat=".2f", dtick=tick_size_price),
        margin=dict(l=40, r=40, t=40, b=40),
        height=600
    )

    chart = fig.to_html(full_html=True)
    historical_data = data.tail(60).to_html()


    return render_template('live-data.html', chart=chart, symbols=symbols, intervals=intervals,
                           selected_stock=selected_stock, selected_interval=selected_interval, historical_data=historical_data, logo_filename = logo_filename)
    



"""@app.route('/download', methods=['POST'])
def download_csv():
    csv_data = request.form['csv_data']
    response = make_response(csv_data)
    response.headers['Content-Disposition'] = 'attachment; filename=historical_data.csv'
    response.headers['Content-type'] = 'text/csv'

    return response
"""
    
"""@app.route('/download', methods=['POST'])
def download():
    # Retrieve the historical data from the session
    
    data = pd.read_html(historical_data)
    data = data.to_excel(data)
    if data:
        # Create a CSV file with the historical data
        csv_filename = 'historical_data.csv'
        with open(csv_filename, 'w') as file:
            file.write(data)

        # Return the CSV file as a download response
        return send_file(csv_filename, as_attachment=True)

    # Return an error message if historical data is not found in the session
    return "No historical data available for download."""

# Define the list of stock symbols
symbols = ['ALTR.LS', 'BCP.LS', 'SLBEN.LS', 'CFN.LS', 'COR.LS', 'CTT.LS', 'EDP.LS', 'EDPR.LS', 'ESON.LS', 'FCP.LS', 'GALP.LS', 'GLINT.LS',
        'GVOLT.LS', 'IBS.LS', 'IPR.LS', 'INA.LS', 'JMT.LS', 'LIG.LS', 'MAR.LS', 'MCP.LS', 'MRL.LS', 'EGL.LS', 'NOS.LS', 'NBA.LS', 'PHR.LS',
        'RAM.LS', 'RED.LS', 'RENE.LS', 'SEM.LS', 'SON.LS', 'SNC.LS', 'SCP.LS', 'TDSA.LS', 'NVG.LS', 'VAF.LS', '^STOXX50E', '^EVZ', 'PSI20.LS']

# Define the recommendation thresholds
buy_threshold = 4
sell_threshold = 4

# Define route for technical indicator
@app.route('/indicators.html', methods=["GET", "POST"])
def technical_indicator():
    logo_filename = "my_logo.png"
    recommendations = {}

    if request.method == 'POST':
        selected_stock = request.form.get('stock')
        selected_term = request.form.get('term')

        stock_data = yf.download(selected_stock, period=selected_term)

        # Calculate recommendations for each indicator
        recommendations = generate_recommendations(stock_data)

    return render_template('indicators.html', logo_filename=logo_filename, symbols=symbols, recommendations=recommendations)


def calculate_rsi(stock_data):
    close_prices = stock_data['Close']
    price_diff = close_prices.diff(1).dropna()
    positive_diff = price_diff[price_diff > 0]
    negative_diff = price_diff[price_diff < 0]
    
    average_gain = positive_diff.rolling(window=14).mean()
    average_loss = abs(negative_diff).rolling(window=14).mean()
    
    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    
    return rsi


def calculate_golden_cross(stock_data):
    short_rolling_mean = stock_data['Close'].rolling(window=50).mean()
    long_rolling_mean = stock_data['Close'].rolling(window=200).mean()
    
    current_price = stock_data['Close'].iloc[-1]
    short_mean = short_rolling_mean.iloc[-1]
    long_mean = long_rolling_mean.iloc[-1]
    
    if short_mean > long_mean and current_price > short_mean:
        return True
    else:
        return False


def calculate_death_cross(stock_data):
    short_rolling_mean = stock_data['Close'].rolling(window=50).mean()
    long_rolling_mean = stock_data['Close'].rolling(window=200).mean()
    
    current_price = stock_data['Close'].iloc[-1]
    short_mean = short_rolling_mean.iloc[-1]
    long_mean = long_rolling_mean.iloc[-1]
    
    if short_mean < long_mean and current_price < short_mean:
        return True
    else:
        return False


def calculate_overbought(stock_data):
    rsi = calculate_rsi(stock_data)
    current_rsi = rsi.iloc[-1]
    
    if current_rsi > 60:
        return True
    else:
        return False


def calculate_oversold(stock_data):
    rsi = calculate_rsi(stock_data)
    current_rsi = rsi.iloc[-1]
    
    if current_rsi < 40:
        return True
    else:
        return False
    
def calculate_macd_line(stock_data, short_period=12, long_period=26):
    # Calculate the short-term exponential moving average (EMA)
    short_ema = stock_data['Close'].ewm(span=short_period, adjust=False).mean()
    
    # Calculate the long-term exponential moving average (EMA)
    long_ema = stock_data['Close'].ewm(span=long_period, adjust=False).mean()
    
    # Calculate the MACD line as the difference between the short-term and long-term EMAs
    macd_line = short_ema - long_ema
    
    return macd_line


def calculate_signal_line(macd_line, signal_period=9):
    # Calculate the signal line as the exponential moving average (EMA) of the MACD line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    return signal_line



def calculate_signal_line_crossover(stock_data):
    macd_line = calculate_macd_line(stock_data)
    signal_line = calculate_signal_line(macd_line)
    
    current_macd_line = macd_line.iloc[-1]
    current_signal_line = signal_line.iloc[-1]
    
    if (current_macd_line > current_signal_line) & (current_macd_line.shift(1) < current_signal_line.shift(1)):
        return True
    else:
        return False


def calculate_stochastic_oscillator(stock_data):
    high_prices = stock_data['High']
    low_prices = stock_data['Low']
    close_prices = stock_data['Close']
    
    lowest_low = low_prices.rolling(window=14).min()
    highest_high = high_prices.rolling(window=14).max()
    
    k_percent = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=3).mean()
    
    stochastic_oscillator = pd.DataFrame({'%K': k_percent, '%D': d_percent})
    
    return stochastic_oscillator


def calculate_high_volume(stock_data):
    volume = stock_data['Volume']
    average_volume = volume.rolling(window=20).mean()
    
    current_volume = volume.iloc[-1]
    current_average_volume = average_volume.iloc[-1]
    
    if current_volume > current_average_volume * 1.5:
        return True
    else:
        return False


def generate_recommendations(stock_data):
    recommendations = {}

    for indicator in ['RSI', 'Golden Cross', 'Death Cross', 'Overbought', 'Oversold', 'Signal Line Crossover', 'Stochastic Oscillator', 'Volume']:
        recommendation = 'Hold'  # Default recommendation is 'Hold'

        if indicator == 'RSI':
            # Implement RSI indicator logic
            if len(stock_data) >= 15:  # Check if enough data points are available
                rsi = calculate_rsi(stock_data)
                current_rsi = rsi.iloc[-1]

                if current_rsi < 40:
                    recommendation = 'Buy'
                elif current_rsi > 60:
                    recommendation = 'Sell'

        elif indicator == 'Golden Cross':
            # Implement Golden Cross indicator logic
            if len(stock_data) >= 200:  # Check if enough data points are available
                golden_cross = calculate_golden_cross(stock_data)
                if golden_cross:
                    recommendation = 'Buy'
                else:
                    recommendation = 'Sell'

        elif indicator == 'Death Cross':
            # Implement Death Cross indicator logic
            if len(stock_data) >= 200:  # Check if enough data points are available
                death_cross = calculate_death_cross(stock_data)
                if death_cross:
                    recommendation = 'Sell'
                else:
                    recommendation = 'Buy'

        elif indicator == 'Overbought':
            # Implement Overbought indicator logic
            if len(stock_data) >= 15:  # Check if enough data points are available
                overbought = calculate_overbought(stock_data)
                if overbought:
                    recommendation = 'Sell'
                else:
                    recommendation = 'Buy'

        elif indicator == 'Oversold':
            # Implement Oversold indicator logic
            if len(stock_data) >= 15:  # Check if enough data points are available
                oversold = calculate_oversold(stock_data)
                if oversold:
                    recommendation = 'Buy'
                else:
                    recommendation = 'Sell'

        elif indicator == 'Signal Line Crossover':
            # Implement Signal Line Crossover indicator logic
            if len(stock_data) >= 26:  # Check if enough data points are available
                signal_line_crossover = calculate_signal_line_crossover(stock_data)
                if signal_line_crossover:
                    recommendation = 'Buy'
                else:
                    recommendation = 'Sell'

        elif indicator == 'Stochastic Oscillator':
            # Implement Stochastic Oscillator indicator logic
            if len(stock_data) >= 14:  # Check if enough data points are available
                stochastic_oscillator = calculate_stochastic_oscillator(stock_data)
                buy_count = sum((stochastic_oscillator['%K'] < 30) & (stochastic_oscillator['%D'] < 30))
                sell_count = sum((stochastic_oscillator['%K'] > 70) & (stochastic_oscillator['%D'] > 70))

                if buy_count > sell_count:
                    recommendation = 'Buy'
                elif sell_count > buy_count:
                    recommendation = 'Sell'

        elif indicator == 'Volume':
            # Implement High Volume indicator logic
            if len(stock_data) >= 20:  # Check if enough data points are available
                high_volume = calculate_high_volume(stock_data)
                if high_volume:
                    recommendation = 'Buy'
                else:
                    recommendation = 'Sell'

        recommendations[indicator] = recommendation
    overall_recommendation = 'Hold'

    # Check if the number of buy recommendations is greater than the sell recommendations
    buy_count = sum([1 for rec in recommendations.values() if rec == 'Buy'])
    sell_count = sum([1 for rec in recommendations.values() if rec == 'Sell'])

    if buy_count > sell_count:
        overall_recommendation = 'Buy'
    elif sell_count > buy_count:
        overall_recommendation = 'Sell'

    recommendations['Advice'] = overall_recommendation
    return recommendations
  

# Define route for market-related news.
@app.route('/news.html', methods=['GET', 'POST'])
def get_market_news():
    logo_filename = "my_logo.png"
    
    company_names = [
    'Select Company Name',
    'Galp Energia, SGPS, S.A.',
    'Altice Portugal, S.A.',
    'Banco Comercial Português, S.A.',
    'Sociedade Lusa de Negócios, SGPS, S.A.',
    'Cofina, SGPS, S.A.',
    'Corticeira Amorim, SGPS, S.A.',
    'CTT - Correios de Portugal, S.A.',
    'EDP - Energias de Portugal, S.A.',
    'EDP Renováveis, S.A.',
    'Efacec Power Solutions, S.A.',
    'F.C. Porto – Futebol, SAD',
    'Global Media Group, SGPS, S.A.',
    'GreenVolt - Energias Renováveis, S.A.',
    'IBS – Indústria e Comércio de Baterias, S.A.',
    'Impresa - Sociedade Gestora de Participações Sociais, S.A.',
    'Inapa – Investimentos, Participações e Gestão, S.A.',
    'Jerónimo Martins, SGPS, S.A.',
    'Laboratórios Inibsa, S.A.',
    'Marshall Monteiro – Investimentos Imobiliários, S.A.',
    'Mota-Engil, SGPS, S.A.',
    'Sonae Capital, SGPS, S.A.',
    'Mota-Engil – Engenharia e Construção, S.A.',
    'NOS, SGPS, S.A.',
    'Novabase – Sistemas de Informação, S.A.',
    'Pharol, SGPS, S.A.',
    'Reditus – Sociedade Gestora de Participações Sociais, S.A.',
    'REN - Redes Energéticas Nacionais, SGPS, S.A.',
    'REN - Renováveis, Energias do Norte, S.A.',
    'Semapa – Sociedade de Investimento e Gestão, SGPS, S.A.',
    'Sonaecom, SGPS, S.A.',
    'Sonae, SGPS, S.A.',
    'Sonae Sierra, SGPS, S.A.',
    'Sociedade Comercial do Plátano, S.A.',
    'TDS – Sociedade de Titularização de Créditos, S.A.',
    'Novabase Valorização de Activos, S.A.',
    'Vanguard Properties, SGPS, S.A.',
    'STOXX Europe 50',
    'VSTOXX®'
    'PSI-20',
    ]

    API_KEY = api_key()
    if request.method == 'POST':
        selected_company = request.form.get('company_name')
    
        
        # Get today's date
        today = datetime.now()
        end = today.strftime('%Y-%m-%d')
        
        # Calculate the date 1 month ago
        one_month_before = today - timedelta(days=28)
        start = one_month_before.strftime('%Y-%m-%d')
        
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': selected_company,
            'from': start,
            'to': end,
            'sortBy': 'publishedAt',
            'apiKey': API_KEY,
        }
        
        response = requests.get(url, params=params)
            
        if response.status_code == 200:
            data = response.json()
            articles = data['articles']
            news = []
            for article in articles:
                news.append(article)
        else:
            print('Request failed with status code:', response.status_code)
        
        return render_template("news.html", logo_filename=logo_filename, company_names=company_names,
                               selected_company=selected_company, news=news, datetime=datetime)
    
    return render_template("news.html", logo_filename=logo_filename, company_names=company_names)



if __name__ == '__main__':
    app.debug = True
    app.run()
