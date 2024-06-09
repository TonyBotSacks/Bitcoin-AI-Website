from flask import Flask, render_template, request
import yfinance as yf
from datetime import datetime
import numpy as np
import keras
from tensorflow.keras.initializers import Orthogonal
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Ensure the static/plots directory exists
os.makedirs('Bitcoin Predictor Website/static/plots', exist_ok=True)

def api(year, month, day, year2, month2, day2):
    ticker = 'BTC-USD'
    bitcoin_data = yf.Ticker(ticker)
    data = bitcoin_data.history(period="3mo")['Close']
    data = np.array(data)
    data = np.reshape(data, (-1, 1))
    date1 = datetime(year, month, day)
    date2 = datetime(year2, month2, day2)
    difference = (date2 - date1).days
    return data, difference

def order(data, scale, division):
    x = []
    y = []
    data = (data / scale).astype("float64")
    for i in range(division, len(data)):
        x.append(data[i-division:i, 0])
        y.append(data[i, 0])
    return x, y

def predict(year, month, day, year2, month2, day2):
    data, difference = api(year, month, day, year2, month2, day2)
    ticker = 'BTC-USD'
    bitcoin_data = yf.Ticker(ticker)

    scale = 20000
    division = 75

    x, y = order(data, scale, division)
    x, y = np.array(x), np.array(y)
    x = x[0]
    x = np.flip(x)
    x = np.reshape(x, (-1, 1))

    custom_objects = {'Orthogonal': Orthogonal}
    model = keras.models.load_model("Bitcoin Predictor Website/Bitcoin AI.h5", custom_objects=custom_objects)
    history = []

    for i in range(difference):
        prediction = model.predict(x).astype("float64")[division-1]
        x = np.delete(x, 0)
        x = np.insert(x, len(x), prediction)
        x = np.reshape(x, (-1, 1))
        prediction = (prediction * scale).astype("float64")
        history.append(prediction)

    return history


@app.route("/")
def home():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        day = int(request.form['day'])
        day2 = int(request.form['day2'])
        month = int(request.form['month'])
        month2 = int(request.form['month2'])
        year = int(request.form['year'])
        year2 = int(request.form['year2'])
        start = f"{year}-{month}-{day}"
        end = f"{year2}-{month2}-{day2}"

        history = predict(year, month, day, year2, month2, day2)
        numbered_list = [i for i in range(1, len(history) + 1)]

        plt.scatter(numbered_list, history, label="Prediction", c="green", alpha=0.7)
        plt.legend()
        plt.title(f"Bitcoin Prediction from {start} to {end}")

        plot_filename = f'plot_{start}_to_{end}.png'
        plot_path = os.path.join('Bitcoin Predictor Website/static/plots', plot_filename)
        plt.savefig(plot_path)
        plt.close()


        return render_template("update.html", plot_filename=plot_filename)
    
@app.route('/restart', methods=['POST'])
def restart():
    return render_template("index.html")



if __name__ == '__main__':
    app.run(debug=True)