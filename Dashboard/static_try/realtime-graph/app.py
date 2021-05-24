from flask import Flask, render_template
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pusher import Pusher
import requests, json, atexit, time, plotly, plotly.graph_objs as go

# create flask app
app = Flask(__name__)

# configure pusher object
pusher = Pusher(
    app_id='YOUR_APP_ID',
    key='YOUR_APP_KEY',
    secret='YOUR_APP_SECRET',
    cluster='YOUR_APP_CLUSTER',
    ssl=True
    )

# define variables for data retrieval
times = []
currencies = ["BTC"]
prices = {"BTC": []}

@app.route("/")
   def index():
       return render_template("index.html")
