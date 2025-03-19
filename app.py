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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Streamlit UI setup
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("\U0001F4C8 Real-time Stock Price Prediction using LSTM")

# API Key Configuration
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
news_api_key = st.secrets["NEWS_API_KEY"]

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, GOOG):", "AAPL").upper()

# User input for prediction days
future_days = st.slider("Select number of days to predict:", min_value=1, max_value=30, value=7)

# Fetch real-time stock data
st.sidebar.subheader("Fetching Data...")
if ticker:
    stock_data = yf.download(ticker, period="2y")

    if stock_data.empty:
        st.sidebar.error("Invalid Ticker! Please enter a valid stock symbol.")
    else:
        st.sidebar.success(f"Data for {ticker} loaded successfully!")
        
        # Data Preprocessing
        stock_data = stock_data[['Close']].copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        stock_data['Scaled_Close'] = scaler.fit_transform(stock_data[['Close']])

        # Prepare data for LSTM
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

        # Reshape for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Build LSTM Model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the Model
        with st.spinner("Training LSTM Model... ⏳"):
            model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=0)

        # Predictions
        Y_pred = model.predict(X_test)
        Y_pred = scaler.inverse_transform(Y_pred.reshape(-1, 1))
        Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

        # Accuracy Metrics
        mse = mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        # Display Metrics
        st.subheader("\U0001F4CA Model Accuracy Metrics")
        st.write(f"\U0001F539 *Mean Squared Error (MSE):* {mse:.4f}")
        st.write(f"\U0001F539 *Mean Absolute Error (MAE):* {mae:.4f}")
        st.write(f"\U0001F539 *R² Score:* {r2:.4f}")

        # Future Prediction
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

        # Display Future Predictions
        future_dates = pd.date_range(stock_data.index[-1] + pd.Timedelta(days=1), periods=future_days)
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_prices.flatten()})
        st.subheader(f"\U0001F4C5 Predicted Stock Prices for Next {future_days} Days")
        st.dataframe(future_df)

        # Plot Actual vs Predicted
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index[-len(Y_test):], y=Y_test.flatten(),
                                 mode='lines', name='Actual Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=stock_data.index[-len(Y_test):], y=Y_pred.flatten(),
                                 mode='lines', name='Predicted Price', line=dict(color='red', dash='dot')))
        fig.update_layout(title=f"Actual vs Predicted Prices ({ticker})",
                          xaxis_title="Date", yaxis_title="Stock Price (USD)")
        st.plotly_chart(fig)

        # Plot Future Predictions
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Close'],
                                        mode='lines+markers', name='Future Predictions', line=dict(color='green')))
        fig_future.update_layout(title=f"Future Stock Price Predictions ({ticker})",
                                 xaxis_title="Date", yaxis_title="Predicted Price (USD)")
        st.plotly_chart(fig_future)

        # ------------------- NEWS + SENTIMENT ------------------
        st.subheader(f"\U0001F4F0 Latest News for {ticker}")

        def fetch_stock_news(stock_name):
            url = f"https://newsapi.org/v2/everything?q={stock_name}&language=en&sortBy=publishedAt&apiKey={news_api_key}"
            response = requests.get(url)
            if response.status_code != 200:
                return []
            articles = response.json().get("articles", [])
            news_data = []
            for article in articles[:5]:
                title = article["title"]
                link = article["url"]
                news_data.append((title, link))
            return news_data

        news_articles = fetch_stock_news(ticker)

        if news_articles:
            all_news_text = "\n".join([title for title, link in news_articles])
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(all_news_text)

            # Gemini Insights
            try:
                gemini_model = genai.GenerativeModel('gemini-pro')
                prompt = f"Summarize the following stock news and provide key takeaways:\n\n{all_news_text}"
                response = gemini_model.generate_content(prompt)
                insights = response.text
            except Exception as e:
                insights = f"Error fetching insights: {e}"

            for title, link in news_articles:
                st.write(f"\U0001F4F0 [{title}]({link})")

            st.subheader("\U0001F4CA Sentiment Analysis:")
            st.write(f"\U0001F539 Compound Score: {sentiment['compound']:.2f}")
            st.write(f"\U0001F539 Positive: {sentiment['pos']*100:.2f}% | Neutral: {sentiment['neu']*100:.2f}% | Negative: {sentiment['neg']*100:.2f}%")
            
            risk_percentage = (1 - (sentiment['compound'] + 1) / 2) * 100
            st.write(f"\u26A0 Risk Percentage: {risk_percentage:.2f}%")

            st.subheader("\U0001F4DD Gemini Insights:")
            st.write(insights)
        else:
            st.warning("No recent news found!")
