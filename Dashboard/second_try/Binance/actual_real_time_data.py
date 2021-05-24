# Importing libraries
from binance.client import Client
import configparser
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor

# Loading keys from config file
config = configparser.ConfigParser()
path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/second_try/Binance/secret.cfg'
config.read_file(open(path))
actual_api_key = config.get('BINANCE', 'ACTUAL_API_KEY')
actual_secret_key = config.get('BINANCE', 'ACTUAL_SECRET_KEY')

client = Client(actual_api_key, actual_secret_key)


def streaming_data_process(msg):
    """
    Function to process the received messages
    param msg: input message
    """
    print(f"message type: {msg['e']}")
    print(f"close price: {msg['c']}")
    print(f"best ask price: {msg['a']}")
    print(f"best bid price: {msg['b']}")
    print("---------------------------")

# Starting the WebSocket
bm = BinanceSocketManager(client)
conn_key = bm.start_symbol_ticker_socket('ETHUSDT', streaming_data_process)
bm.start()
