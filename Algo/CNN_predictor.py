#to find opening next day using already present model
import yfinance as yf
import pickle
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
import tensorflow_estimator
from sklearn.metrics import confusion_matrix
import datetime
today=datetime.date.today()

# to download and fetch market data
def download(startDate,endDate,TimeFrame):

    # Define the ticker symbol for the stock
    tickerSymbol = 'AAPL'
    
    # Fetch the data from Yahoo Finance
    data = yf.download(tickerSymbol, start=startDate, end=endDate, interval=TimeFrame)

    return  data.to_csv(f'CSV/{startDate}_{endDate}_{TimeFrame}')


# Define the start and end date of the data
startDate = '2013-01-01'
endDate = '2023-04-23'

# startDate=today-datetime.timedelta(days=
#                                    )
# endDate = today-datetime.timedelta(days=3)
TimeFrame='1D'

# download(startDate,endDate,TimeFrame)

# Print the data
# print(data)


# Load financial data
data = pd.read_csv(f'E:/Python/Algo/CSV/{startDate}_{endDate}_{TimeFrame}')

startDate = '2023-04-24'
endDate = '2023-05-08'
data2 = pd.read_csv(f'E:/Python/Algo/CSV/{startDate}_{endDate}_{TimeFrame}')# data = pd.read_csv('E:/Python/Algo/CSV/2023-05-01_2023-05-02_1D')
startDate = '2023-05-08'
endDate = '2023-05-28'
data3 = pd.read_csv(f'E:/Python/Algo/CSV/{startDate}_{endDate}_{TimeFrame}')
data=pd.concat((data, data2))
# D:/Python/App_development/Algo/CSV/2023-01-01_2023-03-31_1D.csv
# Preprocess data
X = data.drop(['Date','Open'], axis=1).values
# X=X[:-1]
y = data['Open'].values
y2 = data3['Open'].values
y = np.concatenate((y, y2))
y=y[1:]

####################find model output
filename = 'E:/Python/Algo/Apl_CNN_Model_Open_{startDate}_{endDate}.sav'
# pickle.dump(model, open(filename, 'wb'))
X2 = data3.drop(['Date','Open'], axis=1).values
X = np.concatenate((X, X2))
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)

y_mean = np.mean(y2[1:], axis=0)
y_std = np.std(y2[1:], axis=0)

loaded_model = pickle.load(open(filename, 'rb'))
result=loaded_model.predict(X)
y_pred=abs(result*y_std)+y_mean
df=pd.concat([data,data3])
df.loc[len(df.index)] = None
y_pred1=np.array([[None]])
ynew = np.concatenate((y_pred1, y_pred))
df['y_pred']=None
df['y_pred']=ynew
df['difference']=df['y_pred']-df['Open']
df.to_csv(f'Pred_{startDate}_{endDate}_{TimeFrame}')
