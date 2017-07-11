import quandl

class IDataRetriever:
    def get_data(self):
        pass


class QuandlDataRetriever(IDataRetriever):
    def __init__(self):
        quandl.ApiConfig.api_key = '8_c27qYbGNuH8HAEs5ny'

    def get_data(self, exchange='BITSTAMP', currency='USD'):
        df = quandl.get("BCHARTS/{}{}".format(exchange.upper(), currency.upper()))
        df.columns = ['open', 'high', 'low', 'close', 'volbtc', 'volusd', 'weighted_price']
        print 'Loaded Quandl {}/{} dataset'.format(exchange, currency)
        return df


# class CryptoCompareRetriever(IDataRetriever):
#
#     exchanges = ['Bitstamp', 'Bitfinex', 'Coinbase', 'Kraken', 'Btce', 'Cexio', 'Poloniex', 'Bittrex']
#
#     # exchanges = ['BTCE', 'BTER', 'Bit2C', 'Bitfinex', 'Bitstamp', 'Bittrex', 'CCEDK', 'Cexio', 'Coinbase',
#     #              'Coinfloor', 'Coinse', 'Coinsetter', 'Cryptopia', 'Cryptsy', 'Gatecoin', 'Gemini', 'HitBTC', 'Huobi',
#     #              'itBit', 'Kraken', 'LakeBTC', 'LocalBitcoins', 'MonetaGo', 'OKCoin', 'Poloniex', 'Yacuna', 'Yunbi',
#     #              'Yobit', 'Korbit', 'BitBay', 'BTCMarkets', 'QuadrigaCX', 'CoinCheck', 'BitSquare', 'Vaultoro',
#     #              'MercadoBitcoin', 'Unocoin', 'Bitso', 'BTCXIndia', 'Paymium', 'TheRockTrading', 'bitFlyer', 'Quoine',
#     #              'Luno', 'EtherDelta', 'Liqui', 'bitFlyerFX', 'BitMarket', 'LiveCoin', 'Coinone', 'Tidex', 'Bleutrade',
#     #              'EthexIndia']
#
#     pairs = ['BTC/USD', 'ETH/BTC', 'ETH/USD', 'LTC/USD', 'LTC/BTC']
#
#     # pairs = ['BTC/USD', 'BTC/EUR', 'BTC/ETH', 'ETH/USD', 'ETH/EUR', 'LTC/USD', 'LTC/EUR', 'BTC/LTC']
#
#
#     def get_data(self, from_coin, to_coin, exchange='Bitfinex', limit=10000):
#         params = {
#             'fsym': from_coin,
#             'tsym': to_coin,
#             'e': exchange,
#             'limit': str(limit)
#         }
#         print 'Getting data for {} -> {}/{}...'.format(exchange, from_coin, to_coin)
#         response = requests.get('https://min-api.cryptocompare.com/data/histoday', params)
#         data = json.loads(response.content)['Data']
#         df = pd.DataFrame(data)
#         if len(df.columns) > len(df):
#             df = df.transpose()
#         return df