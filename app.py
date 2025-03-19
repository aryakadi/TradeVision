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
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Streamlit UI setup
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Real-time Stock Price Prediction with News Sentiment Analysis")

# Setup Gemini API from secrets
gemini_api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel('gemini-pro')

# Download NLTK VADER lexicon
nltk.download('vader_lexicon')

# Function to fetch news
def fetch_stock_news(stock_name):
    search_url = f"https://news.google.com/search?q={stock_name}%20stock&hl=en&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    if response.status_code != 200:
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    news_items = soup.find_all('h3')
    news_data = []
    for item in news_items[:5]:
        title = item.text
        link = "https://news.google.com" + item.a['href'][1:]
        news_data.append((title, link))
    return news_data

# Function for sentiment analysis
def analyze_sentiment(news_text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(news_text)
    return sentiment_scores

# Function to summarize news
def get_gemini_insights(news_text):
    prompt = f"Summarize the following stock news and provide key takeaways:\n\n{news_text}"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error fetching insights: {e}"

# User inputs
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, GOOG):", "AAPL").upper()
future_days = st.slider("Select number of days to predict:", min_value=1, max_value=30, value=7)

if ticker:
    # News Section
    st.subheader(f"📰 Latest News for {ticker}")
    news_articles = fetch_stock_news(ticker)
    if news_articles:
        all_news_text = "\n".join([title for title, link in news_articles])
        insights = get_gemini_insights(all_news_text)
        sentiment = analyze_sentiment(all_news_text)

        for title, link in news_articles:
            st.write(f"🔗 [{title}]({link})")

        st.write("📊 **Gemini AI Insights:**")
        st.write(insights)

        st.write("📉 **Sentiment Analysis:**")
        st.write(f"🔹 Compound Score: {sentiment['compound']:.2f}")
        st.write(f"🔹 Positive: {sentiment['pos']*100:.2f}% | Neutral: {sentiment['neu']*100:.2f}% | Negative: {sentiment['neg']*100:.2f}%")
        risk_percentage = (1 - (sentiment['compound'] + 1) / 2) * 100
        st.write(f"⚠ Risk Percentage: {risk_percentage:.2f}%")
    else:
        st.warning("No recent news found!")

    st.divider()
    
    # Stock Data Section
    st.subheader("📈 Stock Price Prediction")
    stock_data = yf.download(ticker, period="2y")
    if stock_data.empty:
        st.error("Invalid Ticker! Please enter a valid stock symbol.")
    else:
        stock_data = stock_data[['Close']].copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        stock_data['Scaled_Close'] = scaler.fit_transform(stock_data[['Close']])

        def create_sequences(data, time_step=50):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 50
        dataset = stock_data['Scaled_Close'].values.reshape(-1, 1)
        X, Y = create_sequences(dataset, time_step)
        train_size = int(len(X) * 0.8)
        X_train, Y_train = X[:train_size], Y[:train_size]
        X_test, Y_test = X[train_size:], Y[train_size:]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        with st.spinner("Training LSTM Model..."):
            model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=0)

        Y_pred = model.predict(X_test)
        Y_pred = scaler.inverse_transform(Y_pred.reshape(-1, 1))
        Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

        mse = mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        st.write(f"🔹 *Mean Squared Error (MSE):* {mse:.4f}")
        st.write(f"🔹 *Mean Absolute Error (MAE):* {mae:.4f}")
        st.write(f"🔹 *R² Score:* {r2:.4f}")

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

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index[-len(Y_test):], y=Y_test.flatten(),
                                 mode='lines', name='Actual Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=stock_data.index[-len(Y_test):], y=Y_pred.flatten(),
                                 mode='lines', name='Predicted Price', line=dict(color='red', dash='dot')))
        fig.update_layout(title=f"Actual vs Predicted Prices ({ticker})",
                          xaxis_title="Date", yaxis_title="Stock Price (USD)")
        st.plotly_chart(fig)

        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Close'],
                                        mode='lines+markers', name='Future Predictions', line=dict(color='green')))
        fig_future.update_layout(title=f"Future Stock Price Predictions ({ticker})",
                                 xaxis_title="Date", yaxis_title="Predicted Price (USD)")
        st.plotly_chart(fig_future)
