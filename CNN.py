import yfinance as yf
import pickle
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
import tensorflow_estimator
from sklearn.metrics import confusion_matrix

# to download and fetch market data
def download(startDate,endDate,TimeFrame):

    # Define the ticker symbol for the stock
    tickerSymbol = 'AAPL'
    
    # Fetch the data from Yahoo Finance
    data = yf.download(tickerSymbol, start=startDate, end=endDate, interval=TimeFrame)

    return  data.to_csv(f'CSV/{startDate}_{endDate}_{TimeFrame}')

import datetime
today=datetime.date.today()
# Define the start and end date of the data
startDate = '2013-01-01'
endDate = '2023-04-23'
startDate=today-datetime.timedelta(days=5)
endDate = today-datetime.timedelta(days=3)
TimeFrame='1D'

download(startDate,endDate,TimeFrame)

# Print the data
# print(data)


# Load financial data
data = pd.read_csv(f'E:/Python/Algo/CSV/{startDate}_{endDate}_{TimeFrame}')
# data = pd.read_csv('E:/Python/Algo/CSV/2023-05-01_2023-05-02_1D')
# D:/Python/App_development/Algo/CSV/2023-01-01_2023-03-31_1D.csv
# Preprocess data
X = data.drop(['Date','Close'], axis=1).values
y_org=data['Close'].values
y = data['Close'].values

# Normalize data

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std
y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)
y = (y - y_mean) / y_std

# Split data into training and testing sets
train_size = int(len(X) * 0.7)
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

# Reshape data for CNN input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# X = np.reshape(X, (X.shape[0], X.shape[1], 1))
# Define CNN architecture
model = Sequential()
model.add(Conv1D(filters=100, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Conv1D(filters=100, kernel_size=2, activation='relu'))
# model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=30, verbose=1)
# model.fit(X_train, y_train, epochs=100, batch_size=170, verbose=1)

# Evaluate model
score = model.evaluate(X_test, y_test, verbose=2)
print('Test Loss:', score)

# Make predictions
y_pred = model.predict(X_test)

y_pred = y_pred * y_std + y_mean
y_test = y_test * y_std + y_mean
print('Difference check:',(y_pred-y_test).mean())

####################find model output
filename = 'Apl_CNN_model.sav'
# pickle.dump(model, open(filename, 'wb'))

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std
y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)
y = (y - y_mean) / y_std
loaded_model = pickle.load(open(filename, 'rb'))
result=loaded_model.predict(X)
print(result*y_std+y_mean)
