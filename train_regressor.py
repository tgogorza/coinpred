import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ggplot import *
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from dataretrievers import QuandlDataRetriever
from transformers.regressor_preprocessor import WindowExtractor

# fix random seed for reproducibility
np.random.seed(123456)
EPOCHS = 300
BATCH_SIZE = 100
SPLIT = 0.80
WINDOW_SIZE = 14

def create_pipeline():

    wnd = WindowExtractor(window_size=WINDOW_SIZE)

    steps = [
        ('wnd', wnd),
        # ('standardize', StandardScaler()),
        # ('standardize', MinMaxScaler()),
        # ('mlp', KerasRegressor(build_fn=create_nn_model, input_dim=X.shape[1], epochs=epochs, batch_size=batch, verbose=2)
        ('mlp', KerasRegressor(build_fn=create_lstm_model, input_dim=wnd.num_features, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2))
    ]
    model = Pipeline(steps)
    return model


def test_model(model, trainX, testX, trainY, testY):
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict))
    print('Train Score: %.2f RMSE' % trainScore)
    testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict))
    print('Test Score: %.2f RMSE' % testScore)

    plot_prediction(np.append(trainY, testY), trainPredict, testPredict)
    plot_pred_vs_actual(testY, testPredict)
    plot_resid(testY, testPredict)

    # df_train = pd.DataFrame([trainY, trainPredict])
    # df_test = pd.DataFrame([testY, testPredict])
    # dfplot = pd.melt(, id_vars=['window'], value_vars=['train_accuracy', 'test_accuracy'])
    # plot = ggplot(dfplot, aes('window', 'value', color='variable')) + geom_line()
    # plot.show()
    # plot = sns.residplot(trainY, trainPredict)
    # plot.show()
    # plot2 = sns.residplot(testY, testPredict)
    # plot2.show()
    # plt.scatter(trainY, trainPredict)
    # plt.show()
    # plt.scatter(testY, testPredict)
    # plt.show()
    return trainScore, testScore


def create_lstm_model(input_dim):
    # create model
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, input_dim), return_sequences=True))
    model.add(LSTM(4))
    model.add(Dense(1))
    model.add(Activation('linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.compile(loss='mse', optimizer='rmsprop')
    return model


def create_nn_model(input_dim):
    model = Sequential()
    print 'Input DIM: {}'.format(input_dim)
    model.add(Dense(units=5000, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=2500, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=500, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def plot_results(df):
    dfplot = pd.melt(df, id_vars=['window'], value_vars=['train_accuracy', 'test_accuracy'])
    plot = ggplot(dfplot, aes('window', 'value', color='variable')) + geom_line()
    plot.show()


def plot_prediction(y, trainPredict, testPredict):
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(y)
    trainPredictPlot[:] = np.nan
    trainPredictPlot = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(y)
    testPredictPlot[:] = np.nan
    # testPredictPlot[len(trainPredict)-(look_back*2)+1:len(dataset)-1, :] = testPredict
    testPredictPlot[len(trainPredict):len(y)] = testPredict
    plt.clf()
    plt.plot(y)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    # plt.savefig('lstm_reg_{}.png'.format(time.time()), dpi=1000)

def plot_pred_vs_actual(testY, testPredict):
    x = testY.reshape(len(testY))
    y = testPredict.reshape(len(testPredict))
    # sns.residplot(x, y, color="g")
    df = pd.DataFrame(np.column_stack([x, y]), columns=['x', 'y'])
    plot = ggplot(df, aes('x', 'y')) + geom_point(color='blue') + stat_smooth(color='blue') + geom_abline(intercept=0, slope=1)
    plot.show()

def plot_resid(testY, testPredict):
    x = testPredict.reshape(len(testPredict))
    y = testY.reshape(len(testY))
    y = (x - y) / y
    # sns.residplot(x, y, lowess=True, color="g")

    df = pd.DataFrame(np.column_stack([x, y]), columns=['x', 'y'])
    plot = ggplot(df[(df.y > -10) & (df.y < 10)], aes('x', 'y')) + geom_point(color='blue') + stat_smooth(color='blue') + geom_abline(intercept=0, slope=1)
    plot.show()

def train_model(X, y):
    print 'Creating Pipeline...'
    model = create_pipeline()
    print 'Training Model...'
    model.fit(X, y)
    # model.fit(X, y[WINDOW_SIZE:])
    # dill.dump(model, 'models/regressor.pkl')
    # model.save('model_classifier_wnd_{}.h5')

    return model


def shape_data(data):
    y_col = 'log_ret_tomorrow'
    X = data.drop([y_col], axis=1).values
    X.astype('float32')
    y = data[y_col].values.reshape(len(data), 1)
    y.astype('float32')

    split = SPLIT
    train_size = int(len(data) * split)
    test_size = len(data) - train_size
    trainX, testX = X[0:train_size, :], X[train_size:len(data), :]
    trainY, testY = y[0:train_size, :], y[train_size:len(data), :]
    print(len(trainX), len(testX))

    # reshape input to be [samples, time steps, features]
    # trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))  # NN
    # testX = np.reshape(testX, (testX.shape[0], testX.shape[1]))  # NN
    # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))  # LSTM
    # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))  # LSTM
    return trainX, trainY, testX, testY


def create_data_set():
    data = QuandlDataRetriever().get_data()
    data['log_ret_tomorrow'] = np.log(data['close'].shift(-1) / data['close'])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    # data = data.ffill().bfill()

    # data['high_tomorrow'] = data.high.shift(-1)
    # data['rel_high_tom'] = data.apply(lambda x: (x['high_tomorrow'] - x['high']) / x['high'], axis=1)
    # data = data.drop(['high_tomorrow'], axis=1)
    # data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data


if __name__ == '__main__':
    data = create_data_set()
    trainX, trainY, testX, testY = shape_data(data)
    model = train_model(trainX, trainY)
    train_score, test_score = test_model(model, trainX, testX, trainY, testY)



