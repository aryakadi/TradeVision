# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# For News and Sentiment Analysis
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# ✅ Add your Gemini API Key here
GEMINI_API_KEY = "AIzaSyCN04ZkiII0AVSu6VUCXsH1wbJwR0bV1IM"  # <-- REPLACE with your Gemini key
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# ------------------------------------
# Streamlit UI setup
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Real-time Stock Price Prediction with Sentiment Analysis")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, GOOG):", "AAPL").upper()

# User input for prediction days
future_days = st.slider("Select number of days to predict:", min_value=1, max_value=30, value=7)

# ------------------------------------
# ✅ Function to Fetch Real-Time News
def fetch_stock_news(stock_name):
    search_url = f"https://news.google.com/search?q={stock_name}%20stock&hl=en&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    if response.status_code != 200:
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    news_items = soup.find_all('h3')

    news_data = []
    for item in news_items[:5]:  # Top 5 news articles
        title = item.text
        link = "https://news.google.com" + item.a['href'][1:]
        news_data.append((title, link))
    
    return news_data

# ✅ Function to Analyze Sentiment
def analyze_sentiment(news_text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(news_text)
    return sentiment_scores

# ✅ Function to Get Insights using Gemini
def get_gemini_insights(news_text):
    prompt = f"Summarize the following stock news and provide key takeaways:\n\n{news_text}"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error fetching insights: {e}"

# ------------------------------------
# ✅ Fetch real-time stock data
if ticker:
    st.sidebar.subheader("Fetching Data...")
    stock_data = yf.download(ticker, period="2y")

    if stock_data.empty:
        st.sidebar.error("Invalid Ticker! Please enter a valid stock symbol.")
    else:
        st.sidebar.success(f"Data for {ticker} loaded successfully!")

        # ✅ Preprocessing Data
        stock_data = stock_data[['Close']].copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        stock_data['Scaled_Close'] = scaler.fit_transform(stock_data[['Close']])

        # ✅ Prepare data for LSTM
        def create_sequences(data, time_step=50):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 50
        dataset = stock_data['Scaled_Close'].values.reshape(-1, 1)
        X, Y = create_sequences(dataset, time_step)

        # Train-test split
        train_size = int(len(X) * 0.8)
        X_train, Y_train = X[:train_size], Y[:train_size]
        X_test, Y_test = X[train_size:], Y[train_size:]

        # Reshape data for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # ✅ Build LSTM Model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # ✅ Train Model
        with st.spinner("Training LSTM Model... ⏳"):
            model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=0)

        # ✅ Predictions
        Y_pred = model.predict(X_test)
        Y_pred = scaler.inverse_transform(Y_pred.reshape(-1, 1))
        Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

        # ✅ Accuracy Metrics
        mse = mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        st.subheader("📊 Model Accuracy Metrics")
        st.write(f"🔹 MSE: {mse:.4f}")
        st.write(f"🔹 MAE: {mae:.4f}")
        st.write(f"🔹 R² Score: {r2:.4f}")

        # ✅ Future Prediction
        def predict_future_prices(model, last_50_days, future_days):
            future_predictions = []
            current_input = last_50_days.reshape(1, -1, 1)

            for _ in range(future_days):
                next_prediction = model.predict(current_input)[0][0]
                future_predictions.append(next_prediction)
                current_input = np.append(current_input[:, 1:, :], [[[next_prediction]]], axis=1)

            return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        last_50_days = dataset[-time_step:]
        future_prices = predict_future_prices(model, last_50_days, future_days)

        future_dates = pd.date_range(stock_data.index[-1] + pd.Timedelta(days=1), periods=future_days)
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_prices.flatten()})
        st.dataframe(future_df)

        # ✅ Fetch and Display News + Sentiment Analysis
        news_articles = fetch_stock_news(ticker)

        if news_articles:
            all_news_text = "\n".join([title for title, link in news_articles])
            insights = get_gemini_insights(all_news_text)
            sentiment = analyze_sentiment(all_news_text)

            # ✅ Risk Calculation
            risk_percentage = (1 - (sentiment['compound'] + 1) / 2) * 100

            st.subheader("📰 Latest News and Sentiment")
            for title, link in news_articles:
                st.markdown(f"🔗 [{title}]({link})")

            st.write(f"📊 *Gemini Insights:* {insights}")
            st.write(f"👍 Positive: {sentiment['pos']*100:.2f}% | 😐 Neutral: {sentiment['neu']*100:.2f}% | 👎 Negative: {sentiment['neg']*100:.2f}%")
            st.write(f"⚠ *Risk Percentage:* {risk_percentage:.2f}%")

# ✅ Run using: streamlit run app.py
