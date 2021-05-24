# Importing libraries
from binance.client import Client
import configparser
import pandas as pd

# Loading keys from config file
config = configparser.ConfigParser()
path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/second_try/Binance/secret.cfg'
config.read_file(open(path))

test_api_key = config.get('BINANCE', 'TEST_API_KEY')
test_secret_key = config.get('BINANCE', 'TEST_SECRET_KEY')
actual_api_key = config.get('BINANCE', 'ACTUAL_API_KEY')
actual_secret_key = config.get('BINANCE', 'ACTUAL_SECRET_KEY')

client = Client(actual_api_key, actual_secret_key)

# Getting earliest timestamp availble (on Binance)
earliest_timestamp = client._get_earliest_valid_timestamp('ETHUSDT', '1d')  # Here "ETHUSDT" is a trading pair and "1d" is time interval
print(earliest_timestamp)

# Getting historical data (candle data or kline)
candle = client.get_historical_klines("ETHUSDT", "1d", earliest_timestamp, limit=1000)

print(candle[1])

eth_df = pd.DataFrame(candle, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
eth_df.dateTime = pd.to_datetime(eth_df.dateTime, unit='ms')
eth_df.closeTime = pd.to_datetime(eth_df.closeTime, unit='ms')
eth_df.set_index('dateTime', inplace=True)
eth_df.to_csv('eth_candle.csv')

print(eth_df.tail(10))
