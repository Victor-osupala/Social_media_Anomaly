import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# Load model and scaler
model = joblib.load("anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Universal Anomaly Detection Dashboard", layout="wide")
st.title("📊 Universal Anomaly Detection Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload any CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully.")

    st.markdown("### 📄 Data Preview")
    st.dataframe(df.head())

    # Detect timestamp column
    timestamp_col = None
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                timestamp_col = col
                break
            except:
                continue

    # Select numeric data
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty:
        st.warning("⚠️ No numeric features found for anomaly detection.")
    else:
        X_scaled = scaler.transform(numeric_df)

        try:
            preds = model.predict(X_scaled)
        except:
            scores = model.decision_function(X_scaled)
            threshold = np.percentile(scores, 5)
            preds = np.where(scores < threshold, 1, 0)

        df["Anomaly"] = preds
        numeric_df["Anomaly"] = preds

        st.metric("Detected Anomalies", int(preds.sum()))
        st.metric("Normal Records", len(preds) - int(preds.sum()))

        # 📊 CHARTS SECTION
        st.markdown("## 📊 Anomaly Analysis Dashboard")

        # 1 & 2. Pie Chart + Histogram
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(df, names="Anomaly", title="Anomaly Distribution",
                        color="Anomaly", hole=0.4,
                        color_discrete_map={0: "green", 1: "red"})
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            hist_col = st.sidebar.selectbox("📊 Histogram Column", numeric_df.columns)
            fig2 = px.histogram(df, x=hist_col, color="Anomaly", barmode="overlay",
                                title=f"{hist_col} Distribution by Anomaly",
                                color_discrete_map={0: "blue", 1: "red"})
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

        # 3 & 4. Scatter Plot + Box Plot
        scatter_cols = st.sidebar.multiselect("🔁 Scatter Plot (Pick 2)", numeric_df.columns, default=numeric_df.columns[:2])
        col3, col4 = st.columns(2)

        with col3:
            if len(scatter_cols) == 2:
                fig3 = px.scatter(df, x=scatter_cols[0], y=scatter_cols[1], color="Anomaly",
                                title=f"{scatter_cols[0]} vs {scatter_cols[1]}",
                                color_discrete_map={0: "green", 1: "red"})
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)

        with col4:
            box_col = st.sidebar.selectbox("📦 Box Plot Column", numeric_df.columns)
            fig4 = px.box(df, x="Anomaly", y=box_col, color="Anomaly",
                        title=f"{box_col} Box Plot by Anomaly")
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)

        # 5 & 6. Correlation + Parallel Coordinates
        col5, col6 = st.columns(2)

        with col5:
            st.markdown("### 🔥 Correlation Matrix")
            fig5 = px.imshow(numeric_df.drop(columns="Anomaly").corr(),
                            text_auto=True, title="Feature Correlation", aspect="auto")
            fig5.update_layout(height=400)
            st.plotly_chart(fig5, use_container_width=True)

        with col6:
            st.markdown("### 📈 Parallel Coordinates")
            par_df = df[numeric_df.columns[:5].tolist()].copy()
            par_df["Anomaly"] = df["Anomaly"]  # Keep numeric
            fig6 = px.parallel_coordinates(par_df,
                                        color="Anomaly",
                                        title="Parallel Coordinates of Features",
                                        color_continuous_scale=[[0, "green"], [1, "red"]],
                                        range_color=[0, 1])
            fig6.update_layout(height=400)
            st.plotly_chart(fig6, use_container_width=True)

        # 🔍 Outlier Explanation
        st.markdown("## 🧠 Outlier Explanation for Top Anomalies")

        zscores = np.abs(zscore(numeric_df.drop(columns="Anomaly")))
        explanation_df = numeric_df[df["Anomaly"] == 1].copy()
        explanation_df["Max Z-Score"] = zscores[df["Anomaly"] == 1].max(axis=1)

        # Show top 5 most anomalous rows
        top_outliers = explanation_df.sort_values("Max Z-Score", ascending=False).head(5)
        st.dataframe(top_outliers.drop(columns=["Anomaly", "Max Z-Score"]))
        st.write("ℹ️ Rows above show the highest deviation in numeric feature space.")

        # 📥 Download option
        st.download_button("⬇️ Download Result CSV", df.to_csv(index=False), "anomaly_results.csv", "text/csv")
else:
    st.info("📥 Upload a CSV file to get started.")
