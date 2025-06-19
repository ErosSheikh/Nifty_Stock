import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px

# --- Phase 1: Data Preprocessing ---
@st.cache_data
def load_data():
    df = pd.read_csv("Nifty_Stocks.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by=['Symbol', 'Date'], inplace=True)
    return df

def create_indicators(df):
    df['SMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(50).mean())
    df['SMA_200'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(200).mean())

    delta = df.groupby('Symbol')['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=12).mean())
    ema26 = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=26).mean())
    df['MACD'] = ema12 - ema26

    df['Volatility'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(20).std())
    return df.dropna()

# --- Phase 2: Machine Learning ---
def train_model(df, model_name):
    features = ['Open', 'High', 'Low', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Volatility']
    df = df.dropna()
    X = df[features]
    y = df['Close']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    if model_name == 'Linear Regression':
        model = LinearRegression()
    elif model_name == 'Random Forest':
        model = RandomForestRegressor()
    else:
        model = XGBRegressor()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    return predictions, y_test, rmse

# --- Phase 3: Streamlit App UI ---
st.set_page_config(page_title="Global Stock Index Predictor", layout="wide")
st.title(" Global Stock Index - Price Prediction App")

df = load_data()
df = create_indicators(df)

symbols = df['Symbol'].unique()
selected_symbol = st.selectbox("Choose a Stock Symbol", symbols)

filtered_df = df[df['Symbol'] == selected_symbol]
st.subheader(f"Recent Data for {selected_symbol}")
st.dataframe(filtered_df.tail(10))

model_name = st.radio("Select Model", ["Linear Regression", "Random Forest", "XGBoost"])
if st.button("Train & Predict"):
    predictions, actuals, rmse = train_model(filtered_df.copy(), model_name)

    results_df = pd.DataFrame({
        'Date': filtered_df.iloc[-len(actuals):]['Date'].values,
        'Actual': actuals.values,
        'Predicted': predictions
    })

    st.subheader(f" Results with {model_name} (RMSE: {rmse:.2f})")
    fig = px.line(results_df, x='Date', y=['Actual', 'Predicted'], title="Actual vs Predicted Close Prices")
    st.plotly_chart(fig, use_container_width=True)
