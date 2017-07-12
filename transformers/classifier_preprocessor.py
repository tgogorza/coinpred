from sklearn.base import TransformerMixin
from stockstats import StockDataFrame
import pandas as pd
import numpy as np

class ClassifierWindowTransformer(TransformerMixin):
    def __init__(self, window_size=1):
        self.window_size = window_size
        # self.num_features = 70
        self.num_features = window_size * 30

    def transform(self, X, *_):

        if type(X) is np.ndarray:
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        df.columns = ['open', 'high', 'low', 'close', 'volbtc', 'volusd', 'weighted_price']
        df = StockDataFrame.retype(df)

        df['rsi_5']
        df['rsi_10']
        df['rsi_15']
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

        df2 = pd.DataFrame()
        for i in xrange(1, self.window_size + 1):
            df['open_-{}_r'.format(i)]
            df['close_-{}_r'.format(i)]
            df['open_-{}_d'.format(i)]
            df['close_-{}_d'.format(i)]
            df['high-{}'.format(i)] = df.high.shift(i)
            df['rsi_5-{}'.format(i)] = df.rsi_5.shift(i)
            df['rsi_10-{}'.format(i)] = df.rsi_10.shift(i)
            df['rsi_15-{}'.format(i)] = df.rsi_15.shift(i)
            df['macd-{}'.format(i)] = df.high.shift(i)
            df['macds-{}'.format(i)] = df.high.shift(i)
            df['macdh-{}'.format(i)] = df.high.shift(i)
            df['cr-{}'.format(i)] = df.high.shift(i)
            df['cr-ma1-{}'.format(i)] = df.high.shift(i)
            df['cr-ma2-{}'.format(i)] = df.high.shift(i)
            df['cr-ma3-{}'.format(i)] = df.high.shift(i)
            df['close_8_sma-{}'.format(i)] = df.high.shift(i)
            df['close_15_sma-{}'.format(i)] = df.high.shift(i)
            df['close_30_sma-{}'.format(i)] = df.high.shift(i)
            df['close_50_sma-{}'.format(i)] = df.high.shift(i)
            df['close_8_ema-{}'.format(i)] = df.high.shift(i)
            df['close_15_ema-{}'.format(i)] = df.high.shift(i)
            df['close_30_ema-{}'.format(i)] = df.high.shift(i)
            df['close_50_ema-{}'.format(i)] = df.high.shift(i)
            df['boll-{}'.format(i)] = df.high.shift(i)
            df['boll_ub-{}'.format(i)] = df.high.shift(i)
            df['boll_lb-{}'.format(i)] = df.high.shift(i)
            df['dma-{}'.format(i)] = df.high.shift(i)
            df['pdi-{}'.format(i)] = df.high.shift(i)
            df['mdi-{}'.format(i)] = df.high.shift(i)
            df['dx-{}'.format(i)] = df.high.shift(i)
            df['adx-{}'.format(i)] = df.high.shift(i)
            df['adxr-{}'.format(i)] = df.high.shift(i)
            df['trix-{}'.format(i)] = df.high.shift(i)
            df['trix_9_sma-{}'.format(i)] = df.high.shift(i)
            df['kdjk-{}'.format(i)] = df.high.shift(i)
            df['kdjd-{}'.format(i)] = df.high.shift(i)
            df['kdjj-{}'.format(i)] = df.high.shift(i)


            # Compute relative high to previous days
            df2['rel_high-{}'.format(i)] = df.apply(
                lambda x: (x['high'] - x['high-{}'.format(i)]) / x['high-{}'.format(i)], axis=1)

            # df2['rel_rsi_5-{}'.format(i)] = df.apply(
            #     lambda x: (x['rsi_5'] - x['rsi_5-{}'.format(i)]) / x['rsi_5-{}'.format(i)], axis=1)
            # df2['rel_rsi_10-{}'.format(i)] = df.apply(
            #     lambda x: (x['rsi_10'] - x['rsi_10-{}'.format(i)]) / x['rsi_10-{}'.format(i)], axis=1)
            # df2['rel_rsi_15-{}'.format(i)] = df.apply(
            #     lambda x: (x['rsi_15'] - x['rsi_15-{}'.format(i)]) / x['rsi_15-{}'.format(i)], axis=1)
            df2['rel_macd-{}'.format(i)] = df.apply(
                lambda x: (x['macd'] - x['macd-{}'.format(i)]) / x['macd-{}'.format(i)], axis=1)
            df2['rel_macds-{}'.format(i)] = df.apply(
                lambda x: (x['macds'] - x['macds-{}'.format(i)]) / x['macds-{}'.format(i)], axis=1)
            df2['rel_macdh-{}'.format(i)] = df.apply(
                lambda x: (x['macdh'] - x['macdh-{}'.format(i)]) / x['macdh-{}'.format(i)], axis=1)
            df2['rel_cr-{}'.format(i)] = df.apply(lambda x: (x['cr'] - x['cr-{}'.format(i)]) / x['cr-{}'.format(i)],
                                                  axis=1)
            df2['rel_cr-ma1-{}'.format(i)] = df.apply(
                lambda x: (x['cr-ma1'] - x['cr-ma1-{}'.format(i)]) / x['cr-ma1-{}'.format(i)], axis=1)
            df2['rel_cr-ma2-{}'.format(i)] = df.apply(
                lambda x: (x['cr-ma2'] - x['cr-ma2-{}'.format(i)]) / x['cr-ma2-{}'.format(i)], axis=1)
            df2['rel_cr-ma3-{}'.format(i)] = df.apply(
                lambda x: (x['cr-ma3'] - x['cr-ma3-{}'.format(i)]) / x['cr-ma3-{}'.format(i)], axis=1)
            df2['rel_close_8_sma-{}'.format(i)] = df.apply(
                lambda x: (x['close_8_sma'] - x['close_8_sma-{}'.format(i)]) / x['close_8_sma-{}'.format(i)], axis=1)
            df2['rel_close_15_sma-{}'.format(i)] = df.apply(
                lambda x: (x['close_15_sma'] - x['close_15_sma-{}'.format(i)]) / x['close_15_sma-{}'.format(i)], axis=1)
            df2['rel_close_30_sma-{}'.format(i)] = df.apply(
                lambda x: (x['close_30_sma'] - x['close_30_sma-{}'.format(i)]) / x['close_30_sma-{}'.format(i)], axis=1)
            df2['rel_close_50_sma-{}'.format(i)] = df.apply(
                lambda x: (x['close_50_sma'] - x['close_50_sma-{}'.format(i)]) / x['close_50_sma-{}'.format(i)], axis=1)
            df2['rel_close_8_ema-{}'.format(i)] = df.apply(
                lambda x: (x['close_8_ema'] - x['close_8_ema-{}'.format(i)]) / x['close_8_ema-{}'.format(i)], axis=1)
            df2['rel_close_15_ema-{}'.format(i)] = df.apply(
                lambda x: (x['close_15_ema'] - x['close_15_ema-{}'.format(i)]) / x['close_15_ema-{}'.format(i)], axis=1)
            df2['rel_close_30_ema-{}'.format(i)] = df.apply(
                lambda x: (x['close_30_ema'] - x['close_30_ema-{}'.format(i)]) / x['close_30_ema-{}'.format(i)], axis=1)
            df2['rel_close_50_ema-{}'.format(i)] = df.apply(
                lambda x: (x['close_50_ema'] - x['close_50_ema-{}'.format(i)]) / x['close_50_ema-{}'.format(i)], axis=1)
            df2['rel_boll-{}'.format(i)] = df.apply(
                lambda x: (x['boll'] - x['boll-{}'.format(i)]) / x['boll-{}'.format(i)], axis=1)
            df2['rel_boll_ub-{}'.format(i)] = df.apply(
                lambda x: (x['boll_ub'] - x['boll_ub-{}'.format(i)]) / x['boll_ub-{}'.format(i)], axis=1)
            df2['rel_boll_lb-{}'.format(i)] = df.apply(
                lambda x: (x['boll_lb'] - x['boll_lb-{}'.format(i)]) / x['boll_lb-{}'.format(i)], axis=1)
            df2['rel_dma-{}'.format(i)] = df.apply(lambda x: (x['dma'] - x['dma-{}'.format(i)]) / x['dma-{}'.format(i)],
                                                   axis=1)
            df2['rel_pdi-{}'.format(i)] = df.apply(lambda x: (x['pdi'] - x['pdi-{}'.format(i)]) / x['pdi-{}'.format(i)],
                                                   axis=1)
            df2['rel_mdi-{}'.format(i)] = df.apply(lambda x: (x['mdi'] - x['mdi-{}'.format(i)]) / x['mdi-{}'.format(i)],
                                                   axis=1)
            df2['rel_dx-{}'.format(i)] = df.apply(lambda x: (x['dx'] - x['dx-{}'.format(i)]) / x['dx-{}'.format(i)],
                                                  axis=1)
            df2['rel_adx-{}'.format(i)] = df.apply(lambda x: (x['adx'] - x['adx-{}'.format(i)]) / x['adx-{}'.format(i)],
                                                   axis=1)
            df2['rel_adxr-{}'.format(i)] = df.apply(
                lambda x: (x['adxr'] - x['adxr-{}'.format(i)]) / x['adxr-{}'.format(i)], axis=1)
            df2['rel_trix-{}'.format(i)] = df.apply(
                lambda x: (x['trix'] - x['trix-{}'.format(i)]) / x['trix-{}'.format(i)], axis=1)
            df2['rel_trix_9_sma-{}'.format(i)] = df.apply(
                lambda x: (x['trix_9_sma'] - x['trix_9_sma-{}'.format(i)]) / x['trix_9_sma-{}'.format(i)], axis=1)
            df2['rel_kdjk-{}'.format(i)] = df.apply(
                lambda x: (x['kdjk'] - x['kdjk-{}'.format(i)]) / x['kdjk-{}'.format(i)], axis=1)
            df2['rel_kdjd-{}'.format(i)] = df.apply(
                lambda x: (x['kdjd'] - x['kdjd-{}'.format(i)]) / x['kdjd-{}'.format(i)], axis=1)
            df2['rel_kdjj-{}'.format(i)] = df.apply(
                lambda x: (x['kdjj'] - x['kdjj-{}'.format(i)]) / x['kdjj-{}'.format(i)], axis=1)

        # Fill NAs
        # df = df.ffill().bfill()
        df2 = df2.ffill().bfill()

        # df2 = df2.replace([np.inf], 10)
        # df2 = df2.replace([-np.inf], -10)
        # df2 = df2.iloc[self.window_size:, :]

        # mat = df.values
        mat = df2.values
        reshaped_mat = np.reshape(mat, (mat.shape[0], 1, mat.shape[-1]))  # LSTM
        # reshaped_mat = np.reshape(mat, (mat.shape[0], mat.shape[-1]))  # NN
        return reshaped_mat

    def fit(self, *_):
        return self


class ClassifierSimpleTransformer(TransformerMixin):
    def __init__(self):
        self.num_features = 38

    def transform(self, X, *_):

        if type(X) is np.ndarray:
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        df.columns = ['open', 'high', 'low', 'close', 'volbtc', 'volusd', 'weighted_price']
        df = StockDataFrame.retype(df)

        df['rsi_5']
        df['rsi_10']
        df['rsi_15']
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

        # Fill NAs
        df = df.ffill().bfill()

        self.num_features = df.shape[-1]

        mat = df.values
        reshaped_mat = np.reshape(mat, (mat.shape[0], 1, mat.shape[-1]))  # LSTM
        # reshaped_mat = np.reshape(mat, (mat.shape[0], mat.shape[-1]))  # NN
        return reshaped_mat

    def fit(self, *_):
        return self