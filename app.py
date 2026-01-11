import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Crypto Price Prediction",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------
# SIDEBAR - CONTROLS + CHATBOT
# -------------------------------------------------
st.sidebar.title("⚙️ Settings")

coin = st.sidebar.selectbox(
    "Select Cryptocurrency",
    ["Bitcoin", "Ethereum", "Solana", "Cardano"]
)

days = st.sidebar.slider(
    "Prediction Days",
    1, 30, 7
)

st.sidebar.markdown("---")

# ---------------- CHATBOT ----------------
st.sidebar.markdown("## 🤖 Crypto Assistant")

def chatbot_reply(question):
    q = question.lower()

    if "price" in q:
        return "Crypto prices depend on market demand, supply, and investor sentiment."
    elif "lstm" in q:
        return "LSTM is a deep learning model used for time-series forecasting."
    elif "predict" in q:
        return "Predictions are generated using historical data patterns."
    elif "accuracy" in q:
        return "Model accuracy is measured using RMSE."
    elif "bitcoin" in q:
        return "Bitcoin is the first and most popular cryptocurrency."
    else:
        return "I can help explain crypto prices, predictions, and ML concepts."

user_question = st.sidebar.text_input("Ask a question")

if user_question:
    st.sidebar.success(chatbot_reply(user_question))

# -------------------------------------------------
# MAIN TITLE
# -------------------------------------------------
st.title("📈 Crypto Price Prediction Dashboard")
st.markdown("Machine Learning based cryptocurrency forecasting web application")

# -------------------------------------------------
# METRICS (DUMMY VALUES)
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

current_price = np.random.randint(30000, 45000)
high_24h = current_price + np.random.randint(500, 1500)
low_24h = current_price - np.random.randint(500, 1500)

col1.metric("Current Price ($)", current_price, "+2.3%")
col2.metric("24h High ($)", high_24h)
col3.metric("24h Low ($)", low_24h)

# -------------------------------------------------
# HISTORICAL DATA (SIMULATED)
# -------------------------------------------------
st.subheader("📊 Historical Price Data")

dates = pd.date_range(end=pd.Timestamp.today(), periods=60)
prices = np.random.randint(30000, 45000, size=60)

df = pd.DataFrame({
    "Date": dates,
    "Price": prices
})

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Price"],
    mode="lines",
    name="Price"
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price ($)",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# DATA PREPROCESSING (ML READY)
# -------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(df[["Price"]])

# Dummy train-test split
train_size = int(len(scaled_prices) * 0.8)
train_data = scaled_prices[:train_size]
test_data = scaled_prices[train_size:]

# Fake predictions (replace with LSTM later)
y_test = test_data.flatten()
y_pred = y_test + np.random.normal(0, 0.02, size=len(y_test))

rmse = math.sqrt(mean_squared_error(y_test, y_pred))

# -------------------------------------------------
# PREDICTION SECTION
# -------------------------------------------------
st.subheader("🔮 Price Prediction")

if st.button("Predict Price"):
    future_price = current_price + np.random.randint(-1000, 1000)

    st.success(
        f"Predicted price after {days} days: **${future_price}**"
    )

    st.info(f"Model RMSE: {rmse:.4f}")

# -------------------------------------------------
# FUTURE PRICE TABLE
# -------------------------------------------------
future_prices = current_price + np.random.randint(-1000, 1000, size=days)

future_df = pd.DataFrame({
    "Day": range(1, days + 1),
    "Predicted Price ($)": future_prices
})

st.subheader("📅 Future Price Forecast")
st.dataframe(future_df)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.markdown(
    "Built with ❤️ using **Streamlit, TensorFlow, Plotly, and Scikit-learn**"
)
