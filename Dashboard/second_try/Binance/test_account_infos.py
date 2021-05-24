# Importing libraries
from binance.client import Client
import configparser

# Loading keys from config file
config = configparser.ConfigParser()
path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/second_try/Binance/secret.cfg'
config.read_file(open(path))

test_api_key = config.get('BINANCE', 'TEST_API_KEY')
test_secret_key = config.get('BINANCE', 'TEST_SECRET_KEY')

client = Client(test_api_key, test_secret_key)

client.API_URL = 'https://testnet.binance.vision/api'  # To change endpoint URL for test account

info = client.get_account()  # Getting account info

balance = client.get_asset_balance(asset='BETH')

print(balance)

