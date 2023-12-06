import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# New dataset
data = pd.DataFrame({
    'Date': pd.to_datetime([
        '2013-01-31', '2013-02-28', '2013-03-31', '2013-04-30', '2013-05-31', '2013-06-30',
        '2013-07-31', '2013-08-31', '2013-09-30', '2013-10-31', '2013-11-30', '2013-12-31',
        '2014-01-31', '2014-02-28', '2014-03-31', '2014-04-30', '2014-05-31', '2014-06-30',
        '2014-07-31', '2014-08-31', '2014-09-30', '2014-10-31', '2014-11-30', '2014-12-31',
        '2015-01-31', '2015-02-28', '2015-03-31', '2015-04-30', '2015-05-31', '2015-06-30',
        '2015-07-31', '2015-08-31', '2015-09-30', '2015-10-31', '2015-11-30', '2015-12-31',
        '2016-01-31', '2016-02-29', '2016-03-31', '2016-04-30', '2016-05-31', '2016-06-30',
        '2016-07-31', '2016-08-31', '2016-09-30', '2016-10-31', '2016-11-30', '2016-12-31',
        '2017-01-31', '2017-02-28', '2017-03-31', '2017-04-30', '2017-05-31', '2017-06-30',
        '2017-07-31', '2017-08-31', '2017-09-30', '2017-10-31', '2017-11-30', '2017-12-31',
        '2018-01-31', '2018-02-28', '2018-03-31', '2018-04-30', '2018-05-31', '2018-06-30',
        '2018-07-31', '2018-08-31', '2018-09-30', '2018-10-31', '2018-11-30', '2018-12-31',
        '2019-01-31', '2019-02-28', '2019-03-31', '2019-04-30', '2019-05-31', '2019-06-30',
        '2019-07-31', '2019-08-31', '2019-09-30', '2019-10-31', '2019-11-30', '2019-12-31',
        '2020-01-31', '2020-02-29', '2020-03-31', '2020-04-30', '2020-05-31', '2020-06-30',
        '2020-07-31', '2020-08-31', '2020-09-30', '2020-10-31', '2020-11-30', '2020-12-31',
        '2021-01-31', '2021-02-28', '2021-03-31', '2021-04-30', '2021-05-31', '2021-06-30',
        '2021-07-31', '2021-08-31', '2021-09-30', '2021-10-31', '2021-11-30', '2021-12-31',
        '2022-01-31', '2022-02-28', '2022-03-31', '2022-04-30', '2022-05-31', '2022-06-30',
        '2022-07-31', '2022-08-31', '2022-09-30', '2022-10-31', '2022-11-30', '2022-12-31',
        '2023-01-31', '2023-02-28', '2023-03-31', '2023-04-30', '2023-05-31', '2023-06-30',
        '2023-07-31', '2023-08-31', '2023-09-30'
    ]),
    'Currency': ['USD'] * 129,
    'Rice_Price': [
        15.30, 15.00, 15.20, 15.40, 15.50, 15.60, 15.80, 15.60, 15.60, 15.40, 15.50, 15.50,
        15.40, 15.40, 15.60, 15.60, 15.40, 15.40, 15.10, 12.80, 12.60, 12.60, 12.60, 12.50,
        12.00, 11.90, 12.30, 12.30, 12.30, 12.80, 13.30, 13.10, 11.80, 11.50, 11.70, 11.70,
        12.10, 11.80, 11.50, 10.80, 11.30, 11.60, 12.00, 12.20, 10.30, 10.30, 10.30, 10.30,
        10.60, 10.10, 10.10, 10.10, 10.20, 10.20, 10.80, 11.30, 11.60, 11.90, 12.00, 11.90,
        11.90, 12.40, 11.80, 13.10, 12.80, 13.10, 13.70, 14.30, 12.50, 11.90, 12.20, 12.90,
        13.50, 12.40, 11.90, 12.00, 11.90, 12.20, 12.90, 12.70, 12.50, 12.40, 12.00, 13.90,
        14.10, 13.80, 14.10, 14.50, 14.90, 14.70, 15.80, 13.70, 13.70, 14.10, 13.80, 14.10,
        14.40, 14.40, 14.40, 15.00, 15.00, 14.60, 15.40, 14.20, 15.00, 14.20, 15.00, 15.40,
        16.10, 16.90, 17.80, 17.70, 17.50, 18.30, 18.50, 18.90, 18.90, 19.40, 19.40, 18.30,
        18.30, 18.50, 18.90, 19.40, 19.80, 19.40, 19.20, 16.60, 16.60
    ]
})

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Normalize the data to the range [0, 1] to improve model training
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Rice_Price']])

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences for training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the predictions to the original scale
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# Calculate MAE and RMSE for training and testing
train_mae = mean_absolute_error(data.iloc[seq_length:train_size], train_predictions)
train_rmse = np.sqrt(mean_squared_error(data.iloc[seq_length:train_size], train_predictions))
test_mae = mean_absolute_error(data.iloc[train_size+seq_length:], test_predictions)
test_rmse = np.sqrt(mean_squared_error(data.iloc[train_size+seq_length:], test_predictions))

print(f"Training MAE: {train_mae:.2f}")
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing MAE: {test_mae:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(data.index[seq_length:train_size], data.iloc[seq_length:train_size], label='Training Data')
plt.plot(data.index[train_size+seq_length:], data.iloc[train_size+seq_length:], label='Testing Data')
plt.plot(data.index[seq_length:train_size], train_predictions, label='Training Predictions')
plt.plot(data.index[train_size+seq_length:], test_predictions, label='Testing Predictions')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Rice Price')
plt.title('Rice Price Prediction with LSTM')
plt.show()
