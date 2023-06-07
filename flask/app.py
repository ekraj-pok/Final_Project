from flask import Flask, render_template, request, session, send_file, Response, make_response
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import datetime
import os
from io import StringIO
import csv
from utils import load_file, localize_tz, additive_decom, calculate_technical_indicators

from keras.models import load_model
from sklearn.metrics import r2_score
import pytz
from pylab import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import load_model




app = Flask(__name__)
app.secret_key = 'Todays_Scession'  # Set the secret key for session management


# define route for home page
@app.route('/')
def home():
    logo_filename = "my_logo.png"
    image_filename = 'project_flow.png'
    return render_template('index.html', image_filename = image_filename, logo_filename = logo_filename)


# Define route for prediction page
@app.route('/prediction.html', methods=['GET', 'POST'])
def predict():
    # Check if a file is uploaded
    logo_filename = "my_logo.png"
    if request.method == 'POST' and 'file' in request.files:
        # Get the uploaded file
        file = request.files['file']
        # Save the file to a specific location
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Load the dataset and perform preprocessing
        df = load_file(file_path)
        df = localize_tz(df)
        df = additive_decom(df)
        df = calculate_technical_indicators(df)

        # Prepare the features for prediction
        X = df[['Open', 'High', 'Low', 'Volume', 'RSI', 'EMA12',
                       'EMA24', 'MACD Line', 'MA12', 'MA24',
                       'BIAS12', 'BIAS24', 'Trend', 'Residual']]
        y = df["target"]

        # Scale the features and target
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        

        # Prepare the input sequences
        window_size = 30
        X_test = []
        for i in range(len(X_scaled) - window_size, len(X_scaled)):
            X_test.append(X_scaled[i - window_size:i])

        X_test = np.array(X_test)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        # Load the saved model
        loaded_model = load_model('../galp_final.h5')

        # Make predictions
        predicted_prices = loaded_model.predict(X_test)
        predicted_prices = pd.DataFrame(scaler.inverse_transform(predicted_prices))

        # Get the last predicted price
        last_predicted_price = predicted_prices.iloc[-1][0]

        # Render the template with the prediction results
        return render_template('prediction.html', predict=True, last_predicted_price=last_predicted_price, logo_filename = logo_filename)

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

    today = datetime.datetime.now().date()
    
    # Set the appropriate period and start date based on the selected interval
    if selected_interval == '1d':  # Daily interval
        period = '2y'
        start = today - datetime.timedelta(days=2*365-1)
    elif selected_interval == '1h':  # Hourly interval
        period = '30d'
        start = today - datetime.timedelta(days=30)
    elif selected_interval == '1m':  # Minute interval
        period = '1d'
        start = today - datetime.timedelta(days=5)
    elif selected_interval == '1wk':  # Weekly interval
        period = 'max'
        #start = today - datetime.timedelta(days=2*365)
    else:  # Monthly interval
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
    """# Add bar trace for volume
    fig.add_trace(go.Bar(x=data.index,
                        y=data['Volume'],
                        name='Volume',
                        marker_color='blue',
                        showlegend=False))
"""
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

    return response"""
"""@app.route('/download', methods=['POST'])
def download():
    # Retrieve the historical data from the session
    
    data = pd.read_html(historical_data)
    
    if data:
        # Create a CSV file with the historical data
        csv_filename = 'historical_data.csv'
        with open(csv_filename, 'w') as file:
            file.write(data)

        # Return the CSV file as a download response
        return send_file(csv_filename, as_attachment=True)

    # Return an error message if historical data is not found in the session
    return "No historical data available for download."""

# Define route for technical indicator
@app.route('/indicators.html', methods = ["GET", "POST"])

def technical_indicator():
    logo_filename = "my_logo.png"

    return render_template('indicators.html', logo_filename = logo_filename)



# Define route for market related news.
@app.route('/news.html', methods = ["GET", "POST"])
def get_market_news():
    logo_filename = "my_logo.png"

    return render_template("news.html", logo_filename = logo_filename)



if __name__ == '__main__':
    app.debug = True
    app.run()
