import pandas as pd
from stockstats import StockDataFrame
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers import Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import quandl

# size=2094
# df = pd.read_csv('data/bitfinexUSD.csv.gz', names=['time', 'close', 'volume'])
quandl.ApiConfig.api_key = '8_c27qYbGNuH8HAEs5ny'
df = quandl.get("BCHARTS/BITSTAMPUSD")
# df = quandl.get("BCHARTS/COINBASEUSD")
df.columns = ['open','high','low','close','volbtc','volusd','weighted_price']
# df.to_csv('btcusd.csv')

df = StockDataFrame.retype(df)
df['rsi_5']
df['rsi_10']
df['rsi_15']
df['open_-1_r']
df['open_-2_r']
df['open_-3_r']
df['open_-4_r']
df['open_-5_r']
df['close_-1_r']
df['close_-2_r']
df['close_-3_r']
df['close_-4_r']
df['close_-5_r']
df['open_-1_d']
df['open_-2_d']
df['open_-3_d']
df['open_-4_d']
df['open_-5_d']
df['close_-1_d']
df['close_-2_d']
df['close_-3_d']
df['close_-4_d']
df['close_-5_d']
df['macd']
df['macds']
df['macdh']
df['cr']
df['cr-ma1']
df['cr-ma2']
df['cr-ma3']
df['close_8_sma']
df['close_15_sma']
df['close_30_sma']
df['close_50_sma']
df['close_8_ema']
df['close_15_ema']
df['close_30_ema']
df['close_50_ema']
df['boll']
df['boll_ub']
df['boll_lb']
df['dma']
df['pdi']
df['mdi']
df['dx']
df['adx']
df['adxr']
df['trix']
df['trix_9_sma']
df['kdjk']
df['kdjd']
df['kdjj']

df['high-1'] = df.high.shift(1)
df['high-2'] = df.high.shift(2)
df['high-3'] = df.high.shift(3)
df['high-4'] = df.high.shift(4)
df['high-5'] = df.high.shift(5)
# df['high-6'] = df.high.shift(6)
# df['high-7'] = df.high.shift(7)
# df['high-8'] = df.high.shift(8)
# df['high-9'] = df.high.shift(9)
# df['high-10'] = df.high.shift(10)
df['high_tomorrow'] = df.high.shift(-1)

df['rel_high_tom'] = df.apply(lambda x: (x['high_tomorrow'] - x['high']) / x['high'], axis=1)
df['rel_high-1'] = df.apply(lambda x: (x['high'] - x['high-1']) / x['high-1'], axis=1)
df['rel_high-2'] = df.apply(lambda x: (x['high'] - x['high-2']) / x['high-2'], axis=1)
df['rel_high-3'] = df.apply(lambda x: (x['high'] - x['high-3']) / x['high-3'], axis=1)
df['rel_high-4'] = df.apply(lambda x: (x['high'] - x['high-4']) / x['high-4'], axis=1)
df['rel_high-5'] = df.apply(lambda x: (x['high'] - x['high-5']) / x['high-5'], axis=1)
# df['rel_high-6'] = df.apply(lambda x: (x['high'] - x['high-6']) / x['high-6'], axis=1)
# df['rel_high-7'] = df.apply(lambda x: (x['high'] - x['high-7']) / x['high-7'], axis=1)
# df['rel_high-8'] = df.apply(lambda x: (x['high'] - x['high-8']) / x['high-8'], axis=1)
# df['rel_high-9'] = df.apply(lambda x: (x['high'] - x['high-9']) / x['high-9'], axis=1)
# df['rel_high-10'] = df.apply(lambda x: (x['high'] - x['high-10']) / x['high-10'], axis=1)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df['rise'] = df.rel_high_tom.apply(lambda x: 1.0 if x > 0.0 else 0.0)


# df = df.iloc[1:-1,:]
# df = df.iloc[5:,:]

# plt = ggplot(df.tail(1000), aes('time', 'close')) + geom_line()

# def get_change_lvl(rate):
#     sign = 1 if rate >= 0 else -1
#     rate = abs(rate)
#     # level = divmod(rate, 0.5)[0]
#     level = 0
#     if rate > 10:
#         level = 5
#     elif rate > 5:
#         level = 4
#     elif rate > 3:
#         level = 3
#     elif rate > 2:
#         level = 2
#     elif rate > 1:
#         level = 1
#     return level * sign

# df['rel_change'] = df.apply(lambda x: float(x['close']-x['open'])/x['open'], axis=1)
# df['rise_lvl'] = df['change'].fillna(0.0).apply(get_change_lvl)

df.to_csv('btcusd_proc.csv')

# fix random seed for reproducibility
np.random.seed(7)

# dataset = df.close.values.reshape(len(df), 1)
# dataset.astype('float32')
# normalize the dataset

# dataset = df.values
# dataset.astype('float32')

# X = df[['open','close_-1','close_-2','close_-3','close_-4','close_-5']].values
# X = df[['open','high','low','close_-1','close_-2','close_-3','close_-4','close_-5','volbtc','volusd','weighted_price']].values
# X = df[['open','high','low','close','volbtc','volusd','weighted_price']].values
# X = df[['volbtc','volusd','rel_high-2','rel_high-3','rel_high-4','rel_high-5','rel_high-6']].values
y_col = 'rise'
X = df.drop([y_col, 'high_tomorrow', 'rel_high_tom'], axis=1).values
# X = df[['rel_high-1', 'rel_high-2', 'rel_high-3', 'rel_high-4', 'rel_high-5', 'rel_high-6', 'rel_high-7', 'rel_high-8', 'rel_high-9', 'rel_high-10']].values
X.astype('float32')
y = df[y_col].values.reshape(len(df), 1)
y.astype('float32')

# x_scaler = MinMaxScaler(feature_range=(0, 1))
# X = x_scaler.fit_transform(X)
# y_scaler = MinMaxScaler(feature_range=(0, 1))
# y = y_scaler.fit_transform(y)
# joblib.dump(y_scaler, 'scaler.pkl')

train_size = int(len(df) * 0.85)
test_size = len(df) - train_size
trainX, testX = X[0:train_size, :], X[train_size:len(df), :]
trainY, testY = y[0:train_size, :], y[train_size:len(df), :]
print(len(trainX), len(testX))


def create_lstm_model():
    # create model
    model = Sequential()
    model.add(LSTM(100, input_shape=(trainX.shape[0], trainX.shape[1])))      # 42
    # model.add(LSTM(30, return_sequences=True))      # 42
    # model.add(LSTM(5))      # 42
    # model.add(Dense(25))
    # model.add(Dropout(0.2))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def create_lstm_classifier_model():
    # create model
    model = Sequential()
    model.add(LSTM(100, input_shape=(1, trainX.shape[2])))      # 42
    model.add(Dropout(0.5))
    # model.add(LSTM(10000))
    # model.add(Dropout(0.5))
    # model.add(LSTM(5000))
    # model.add(Dropout(0.5))
    # model.add(LSTM(1000))
    # model.add(Dropout(0.5))
    # model.add(LSTM(100))
    # model.add(Dense(25))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_nn_model(with_drop=False):
    model = Sequential()
    model.add(Dense(units=500, input_dim=trainX.shape[1], activation='relu'))
    # model.add(Dense(units=500, input_shape=(1, trainX.shape[2]), activation='relu'))
    model.add(Dense(units=250, activation='relu'))
    if with_drop:
        model.add(Dropout(0.25))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def create_nn_classifier_model():
    model = Sequential()
    model.add(Dense(units=1000, input_dim=trainX.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=5000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_nn(epochs, with_drop=False):
    batch_size = 1
    model = create_nn_model(with_drop)
    model.save('model_nn_ep{}_{}.h5'.format(epochs,'drop' if with_drop else ''))
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)
    return model

def train_lstm(epochs):
    batch_size = 1
    model = create_lstm_model()
    model.save('model_lstm_ep{}.h5'.format(epochs))
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)
    return model

def train_nn_classifier(epochs):
    batch_size = 1
    model = create_nn_classifier_model()
    model.save('model_nn_classifier_ep{}.h5'.format(epochs))
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)
    return model

def train_lstm_classifier(epochs):
    batch_size = 1
    model = create_lstm_classifier_model()
    model.save('model_lstm_ep{}.h5'.format(epochs))
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)
    return model

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))  # NN
testX = np.reshape(testX, (testX.shape[0], testX.shape[1]))      # NN
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))    # LSTM
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))        # LSTM


epochs = 250
steps = [
        ('standardize', StandardScaler()),
        # ('lstm', KerasClassifier(build_fn=create_lstm_classifier_model, epochs=epochs, batch_size=100, verbose=2))
        ('mlp', KerasClassifier(build_fn=create_nn_classifier_model, epochs=epochs, batch_size=200, verbose=2))
        ]
model = Pipeline(steps)
model.fit(trainX, trainY)

# model = train_nn_classifier(5)
# model = train_lstm_classifier(100)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % trainScore)
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % testScore)


# trainPred = map(lambda x: 1 if x >= 0.5 else 0, trainPredict)
print('Train Accuracy: {}'.format(accuracy_score(trainY, trainPredict)))
print confusion_matrix(trainY, trainPredict)
# testPred = map(lambda x: 1 if x >= 0.5 else 0, testPredict)
print('Test Accuracy: {}'.format(accuracy_score(testY, testPredict)))
print confusion_matrix(testY, testPredict)