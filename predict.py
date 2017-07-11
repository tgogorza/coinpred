from keras import models
from sklearn.externals import joblib
import quandl

quandl.ApiConfig.api_key = '8_c27qYbGNuH8HAEs5ny'
df = quandl.get("BCHARTS/BITSTAMPUSD")
df.columns = ['open','high','low','close','volbtc','volusd','weighted_price']

input = df.tail(1)[['open','high','low','close','volbtc','volusd','weighted_price']].values
print input
input = input.reshape((input.shape[0], 1, input.shape[1]))
model = models.load_model('model_window_30.h5')
pred_high = model.predict(input)
scaler = joblib.load('scaler.pkl')
pred_high = scaler.inverse_transform(pred_high)
print pred_high