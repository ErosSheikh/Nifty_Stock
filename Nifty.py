import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import datetime as dt

# --- Page Config ---
st.set_page_config(page_title="Nifty Stock Analyzer", layout="wide")

st.title(" Nifty Stocks Technical Analysis Dashboard")

# --- Load CSV ---
@st.cache_data
def load_data():
    df = pd.read_csv("Nifty_Stocks.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    return df

df = load_data()

# --- Sidebar Filters ---
stocks = df['Symbol'].unique()
selected_stock = st.sidebar.selectbox("Choose a stock", stocks)

# Filter Data
stock_df = df[df['Symbol'] == selected_stock].copy()
stock_df.set_index('Date', inplace=True)

# --- Technical Indicators ---
stock_df['SMA_50'] = stock_df['Close'].rolling(window=50).mean()
stock_df['SMA_200'] = stock_df['Close'].rolling(window=200).mean()

# RSI
delta = stock_df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
stock_df['RSI'] = 100 - (100 / (1 + rs))

# MACD
ema12 = stock_df['Close'].ewm(span=12, adjust=False).mean()
ema26 = stock_df['Close'].ewm(span=26, adjust=False).mean()
stock_df['MACD'] = ema12 - ema26
stock_df['Signal'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()

# Volatility
stock_df['Volatility'] = stock_df['Close'].pct_change().rolling(window=14).std()

# --- Plotting ---
st.subheader(f" Price & SMA - {selected_stock}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['SMA_50'], mode='lines', name='SMA 50'))
fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['SMA_200'], mode='lines', name='SMA 200'))
st.plotly_chart(fig, use_container_width=True)

# --- RSI Plot ---
st.subheader(" Relative Strength Index (RSI)")
rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['RSI'], name="RSI"))
rsi_fig.update_layout(yaxis_range=[0, 100])
st.plotly_chart(rsi_fig, use_container_width=True)

# --- MACD Plot ---
st.subheader(" MACD")
macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MACD'], name="MACD"))
macd_fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Signal'], name="Signal"))
st.plotly_chart(macd_fig, use_container_width=True)

# --- Volatility Plot ---
st.subheader("Volatility (Standard Deviation)")
vol_fig = go.Figure()
vol_fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Volatility'], name="Volatility"))
st.plotly_chart(vol_fig, use_container_width=True)

# --- Download Option ---
st.download_button(
    " Download Data as CSV",
    stock_df.reset_index().to_csv(index=False),
    file_name=f"{selected_stock}_analysis.csv"
)
