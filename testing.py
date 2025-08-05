import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import datetime
import re

# === App Title ===
st.set_page_config(page_title="Multi-Anomaly Detection System", layout="wide")
st.title("🔍 Social Media Anomaly Detection")

# === Sidebar Navigation ===
option = st.sidebar.radio("Navigate", ['Outlier Detection', 'Time-Series Anomaly', 'Textual Anomaly Detection', 'Sentiment Analysis'])

# === Load Dataset for applicable tabs ===
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# === Text Preprocessing ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# === Outlier Detection ===
if option == "Outlier Detection":
    st.header("📌 Outlier Detection using Isolation Forest")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.subheader("Dataset Overview")
        st.write(df.head())

        # Select only numerical columns
        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] > 0:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)

            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit(scaled_data)
            df['Outlier'] = model.predict(scaled_data)

            st.success("Outlier Detection Complete ✅")
            st.write(df['Outlier'].value_counts())

            # Visualize result
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=df, x=numeric_df.columns[0], y=numeric_df.columns[1], hue='Outlier', palette='coolwarm', ax=ax)
            st.pyplot(fig)

            st.download_button("Download Labeled Data", df.to_csv(index=False), file_name="labeled_data.csv")
        else:
            st.warning("No numerical columns found in the uploaded file.")

# === Time-Series Anomaly Detection ===
elif option == "Time-Series Anomaly":
    st.header("📊 Time-Series Anomaly Detection")

    uploaded_file = st.file_uploader("Upload a CSV file with timestamp column", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.subheader("Dataset Sample")
        st.write(df.head())

        timestamp_col = st.selectbox("Select Timestamp Column", df.columns)
        value_col = st.selectbox("Select Value Column (Numeric)", df.select_dtypes(include=np.number).columns)

        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values(by=timestamp_col)

            # Detect anomaly
            model = IsolationForest(contamination=0.03)
            df['anomaly'] = model.fit_predict(df[[value_col]])
            df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

            # Plot
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df[timestamp_col], df[value_col], label='Value')
            ax.scatter(df[df['anomaly'] == 1][timestamp_col], df[df['anomaly'] == 1][value_col], color='red', label='Anomaly')
            ax.set_title("Time-Series Anomalies")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

# === Textual Anomaly Detection ===
elif option == "Textual Anomaly Detection":
    st.header("🧠 Textual Anomaly Detection")
    user_text = st.text_area("Enter Text for Anomaly Detection")

    if st.button("Detect Anomaly"):
        if user_text.strip():
            cleaned = clean_text(user_text)
            word_count = len(cleaned.split())
            unusual = word_count < 3 or word_count > 100  # crude textual anomaly check

            if unusual:
                st.error("⚠️ Text is Anomalous")
            else:
                st.success("✅ Text is Normal")
        else:
            st.warning("Please enter text.")

# === Sentiment Analysis ===
elif option == "Sentiment Analysis":
    st.header("💬 Sentiment Analysis")
    user_input = st.text_area("Enter text to analyze sentiment:")

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            blob = TextBlob(user_input)
            sentiment = blob.sentiment.polarity

            if sentiment > 0:
                st.success(f"🙂 Positive Sentiment (Score: {sentiment:.2f})")
            elif sentiment < 0:
                st.error(f"🙁 Negative Sentiment (Score: {sentiment:.2f})")
            else:
                st.info(f"😐 Neutral Sentiment (Score: {sentiment:.2f})")
        else:
            st.warning("Text input is empty.")
