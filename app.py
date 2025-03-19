import streamlit as st
import pandas as pd
import requests

st.title("📈 TradeVusion - Stock Price & Sentiment Analysis")

stock = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):")

if stock:
    st.write(f"Showing analysis for: **{stock.upper()}**")

    # Predicted Price Placeholder
    st.write("🔮 Predicted Price: $XXX.XX")

    # News Sentiment Section
    st.subheader("📰 Recent News Sentiment")
    api_key = st.secrets.get("NEWS_API_KEY", "YOUR_API_KEY")
    url = f'https://newsapi.org/v2/everything?q={stock}&sortBy=publishedAt&apiKey={api_key}'
    response = requests.get(url)
    articles = response.json().get('articles', [])
    
    for article in articles[:5]:
        st.markdown(f"**{article['title']}**")
        st.write(article['description'])
        st.write(f"[Read more]({article['url']})")
        st.write("---")
