import numpy as np
import keras
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, LSTM, Dropout, Activation, BatchNormalization, LeakyReLU
import keras.regularizers as regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from dataretrievers import QuandlDataRetriever
from transformers.classifier_preprocessor import ClassifierSimpleTransformer, ClassifierWindowTransformer

np.random.seed(123456)
EPOCHS = 300
BATCH_SIZE = 100
SPLIT = 0.80
WINDOW_SIZE = 7


def create_lstm_classifier_model(input_dim):
    # create model
    model = Sequential()
    model.add(LSTM(100, input_shape=(1, input_dim), return_sequences=True))
    # model.add(LSTM(32, return_sequences=True, stateful=True,
    #                batch_input_shape=(BATCH_SIZE, timesteps, data_dim)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(7))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_nn_classifier_model(input_dim):
    model = Sequential()
    model.add(Dense(1000, input_dim=input_dim, activity_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(500, activity_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(100, activity_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_pipeline():

    # preproc = ClassifierSimpleTransformer()
    preproc = ClassifierWindowTransformer(window_size=WINDOW_SIZE)
    steps = [
            ('preproc', preproc),
            # ('standardize', StandardScaler()),
            # ('lstm', KerasClassifier(build_fn=create_lstm_classifier_model, input_dim=preproc.num_features, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2))
            ('mlp', KerasClassifier(build_fn=create_lstm_classifier_model, input_dim=preproc.num_features, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2))
            ]
    model = Pipeline(steps)
    return model


def train_model(X, y):
    print 'Creating Pipeline...'
    model = create_pipeline()
    print 'Training Model...'
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
    history = model.fit(X, y)

    # plt.figure()
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='best')

    # model.fit(X, y[WINDOW_SIZE:])
    # dill.dump(model, 'models/regressor.pkl')
    # model.save('model_classifier_wnd_{}.h5')

    return model


def shape_data(data):
    y_col = 'rise'
    X = data.drop([y_col], axis=1).values
    X.astype('float32')
    y = data[y_col].values.reshape(len(data), 1)
    # y.astype('float32')

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
    data = data.replace(0.0, np.NaN).ffill()
    data['rel_ret_tom'] = (data.close.shift(-1) - data.close) / data.close
    # data['log_ret_tomorrow'] = np.log(data['close'].shift(-1) / data['close'])
    data['rise'] = data.rel_ret_tom.apply(lambda x: 1 if x > 0.0 else 0)
    # data['rise'] = keras.utils.to_categorical(data['rise'], num_classes=2)
    data = data.drop('rel_ret_tom', axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    # data = data.ffill().bfill()

    # data['high_tomorrow'] = data.high.shift(-1)
    # data['rel_high_tom'] = data.apply(lambda x: (x['high_tomorrow'] - x['high']) / x['high'], axis=1)
    # data = data.drop(['high_tomorrow'], axis=1)
    # data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data


def test_model(model, trainX, testX, trainY, testY):
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # trainPred = map(lambda x: 1 if x >= 0.5 else 0, trainPredict)
    print('Train Accuracy: {}'.format(accuracy_score(trainY, trainPredict)))
    print confusion_matrix(trainY, trainPredict)
    # testPred = map(lambda x: 1 if x >= 0.5 else 0, testPredict)
    print('Test Accuracy: {}'.format(accuracy_score(testY, testPredict)))
    print confusion_matrix(testY, testPredict)

    # plot_prediction(np.append(trainY, testY), trainPredict, testPredict)
    # plot_pred_vs_actual(testY, testPredict)
    # plot_resid(testY, testPredict)

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
    # return trainScore, testScore


if __name__ == '__main__':
    data = create_data_set()
    trainX, trainY, testX, testY = shape_data(data)
    model = train_model(trainX, trainY)
    test_model(model, trainX, testX, trainY, testY)
