import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math

st.set_page_config(page_title="Crypto LSTM Dashboard", layout="wide")
st.title("🚀 Cryptocurrency Price Prediction")

# ----------------------------
# Fetch Crypto Data
# ----------------------------
@st.cache_data
def get_crypto_data(coin="bitcoin", days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        st.error("❌ Failed to fetch data from CoinGecko")
        st.stop()

    prices = response.json()["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# ----------------------------
# Sidebar
# ----------------------------
coins = ["bitcoin", "ethereum", "cardano", "dogecoin", "solana"]
selected_coin = st.sidebar.selectbox("Select Cryptocurrency", coins)

# ----------------------------
# Load Data
# ----------------------------
data = get_crypto_data(selected_coin)

st.subheader(f"{selected_coin.capitalize()} Price History")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=data["timestamp"],
    y=data["price"],
    mode="lines",
    name="Price"
))
fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# LSTM Preparation
# ----------------------------
dataset = data["price"].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

def create_sequences(data, time_step=10):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 10
X, y = create_sequences(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ----------------------------
# Train Model (Cached)
# ----------------------------
@st.cache_resource
def train_model(X, y, time_step):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    return model

model = train_model(X, y, time_step)

# ----------------------------
# Model Evaluation
# ----------------------------
train_pred = model.predict(X, verbose=0)
rmse = math.sqrt(mean_squared_error(y, train_pred))
st.info(f"📉 Model RMSE: {rmse:.4f}")

# ----------------------------
# Predict Next 7 Days
# ----------------------------
temp_input = scaled_data[-time_step:].reshape(1, time_step, 1)
predictions = []

for _ in range(7):
    pred = model.predict(temp_input, verbose=0)
    predictions.append(pred[0, 0])
    temp_input = np.concatenate(
        (temp_input[:, 1:, :], pred.reshape(1, 1, 1)),
        axis=1
    )

pred_prices = scaler.inverse_transform(
    np.array(predictions).reshape(-1, 1)
)

future_dates = pd.date_range(
    start=data["timestamp"].iloc[-1] + pd.Timedelta(days=1),
    periods=7
)

pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price (USD)": pred_prices.flatten()
})

st.subheader("📅 Next 7 Days Prediction")
st.table(pred_df)

# ----------------------------
# Decision Logic
# ----------------------------
current_price = float(data["price"].iloc[-1])
final_price = float(pred_prices[-1])

change_percent = ((final_price - current_price) / current_price) * 100

if change_percent > 5:
    decision = "BUY"
elif change_percent < -5:
    decision = "SELL"
else:
    decision = "HOLD"

st.subheader("📊 Investment Suggestion")

if decision == "BUY":
    st.success("🟢 BUY – Expected price increase")
elif decision == "SELL":
    st.error("🔴 SELL – Expected price decrease")
else:
    st.warning("🟡 HOLD – Market stable")

# ----------------------------
# Prediction Chart
# ----------------------------
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=data["timestamp"],
    y=data["price"],
    mode="lines",
    name="Historical Price"
))

fig2.add_trace(go.Scatter(
    x=pred_df["Date"],
    y=pred_df["Predicted Price (USD)"],
    mode="lines+markers",
    name="Predicted Price"
))

fig2.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig2, use_container_width=True)
