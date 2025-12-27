import streamlit as st
import pandas as pd
from promotion_predictor import predict_promotion
st.set_page_config(page_title="Promotion Predictor", layout="centered")
st.title("🎯 Promotion Eligibility Predictor")

uploaded_file = st.file_uploader("📤 Upload Employee Data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df.head())

    if st.button("🔍 Predict Promotions"):
        preds, probs = predict_promotion(df)
        df['Promotion_Probability'] = probs
        df['Promotion_Prediction'] = preds
        st.success("✅ Predictions completed!")
        st.subheader("📊 Results")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", data=csv, file_name='promotion_predictions.csv', mime='text/csv')