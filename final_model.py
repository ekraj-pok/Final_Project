# import modules
import pandas as pd
import numpy as np
import pytz

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score

from statsmodels.tsa.seasonal import seasonal_decompose

from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Conv1D, MaxPooling1D, LSTM, Dense, Flatten
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from keras.models import load_model

from utils import load_file, localize_tz, additive_decom, calculate_technical_indicators,split_data, fit_scaler

from scikeras.wrappers import KerasRegressor

df = load_file("GALP.LS_daily_data.xlsx")
df = localize_tz(df)
df = additive_decom(df)
df = calculate_technical_indicators(df)
X, y = split_data(df)
X_scaled, y_scaled = fit_scaler(X, y)



# sequence of window for input
window_size = 20
# Create input sequences and corresponding labels
X_sequence, y_sequence = [], []

for i in range(window_size, len(X_scaled)):
    X_sequence.append(X_scaled[i-window_size:i])
    y_sequence.append(y_scaled[i])  

# Convert the data to numpy arrays
X_sequence, y_sequence = np.array(X_sequence), np.array(y_sequence)


# Reshape the data for input to the CNN-LSTM model
X_sequence = X_sequence.reshape((X_sequence.shape[0], X_sequence.shape[1], X_sequence.shape[2]))

# Build the model (best paremeter: filter_size': 64, 'kernel_size': 3, 'pool_size': 2, 'unit': 150, 'epoch': 10, 'batch_size': 20)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, X_sequence.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(125, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# Train the model
model.fit(X_sequence, y_sequence, epochs=13, batch_size=20)

# Predict the model on the test set
predicted_prices = model.predict(X_sequence)
# Inverse transform the predicted prices
predicted_prices = fit_scaler.scaler.inverse_transform(predicted_prices)
pred_df = pd.DataFrame(predicted_prices)
# Inverse transform the actual prices
actual_prices = fit_scaler.scaler.inverse_transform(y_sequence.reshape(-1, 1))

# Claculate MSE
error = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
r2 = r2_score(actual_prices, predicted_prices)


# Plot the desired output and predicted prices against the dates
plt.plot(X_sequence[window_size:].index, actual_prices, label='Closing Price')
plt.plot(X_sequence[window_size:].index, predicted_prices, label='Predicted Prices')

# Add labels and title to the plot
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.title('Closing Price vs. Predicted Prices')
plt.show()
print(f"RMSE for test data is: {error}")
print(f"r2 for test data is : {r2}")
print(f"predicted price for tomorrow based on todays date of {X_sequence[20:].index[-1]} is {pred_df[0][-1:]}")