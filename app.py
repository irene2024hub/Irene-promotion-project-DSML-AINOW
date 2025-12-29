

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

from predict_promotion import predict_promotion


# LOAD MODEL (FOR SHAP)
# -----------------------------------
model = joblib.load("xgb_promotion_model.pkl")


# LOAD DEFAULT DATASET
# -----------------------------------
@st.cache_data
def load_dataset():
    return pd.read_csv("promotion_dataset.csv")

base_df = load_dataset()

st.set_page_config(page_title="Promotion Predictor", layout="centered")

st.title("Promotion Eligibility Predictor")
st.caption("Promoted = 1 | Not Promoted = 0")

# -----------------------------------
# HR ACCESS CONTROL
# -----------------------------------
st.sidebar.header("Access Control")
is_hr = st.sidebar.checkbox("I am HR / Management")

if is_hr:
    threshold = st.sidebar.slider(
        "Promotion Threshold",
        min_value=0.30,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Higher value = stricter promotion decision"
    )
else:
    threshold = 0.50

st.sidebar.caption(f"Current Threshold: {threshold}")


# TABS

tab1, tab2 = st.tabs(["Dataset & Bulk Prediction", "Test Single Employee"])

# TAB 1: DATASET + BULK CSV
# -----------------------------------
with tab1:
    st.subheader("Promotion Dataset Preview")
    st.caption("Default dataset used during model development")

    st.dataframe(base_df.head())

    uploaded_file = st.file_uploader(
        "Upload Employee Data (CSV) – Optional",
        type="csv"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Custom dataset uploaded")
    else:
        df = base_df.copy()

    if st.button("Predict Promotions"):
        preds, probs = predict_promotion(df, threshold)

        df["Promotion_Probability"] = probs
        df["Promotion_Prediction"] = preds

        st.success("Predictions completed")
        st.dataframe(df.head(20))

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results as CSV",
            data=csv,
            file_name="promotion_predictions.csv",
            mime="text/csv"
        )


# TAB 2: SINGLE EMPLOYEE TEST FORM
with tab2:
    st.subheader("Test Promotion for a Single Employee")
    st.caption("Use realistic values based on employee records")

    with st.form("promotion_form"):
        Trainings_Attended = st.number_input(
            "Trainings Attended", 0, step=1,
            help="Total number of formal trainings completed"
        )

        Year_of_birth = st.number_input(
            "Year of Birth", 1950, 2010,
            help="Employee year of birth (YYYY)"
        )

        Last_performance_score = st.slider(
            "Last Performance Score", 0.0, 5.0, 3.0,
            help="Most recent performance rating"
        )

        Year_of_recruitment = st.number_input(
            "Year of Recruitment", 1990, 2025,
            help="Year employee joined the organization"
        )

        Targets_met = st.selectbox(
            "Targets Met", [0, 1],
            help="1 = Met targets, 0 = Did not meet targets"
        )

        Previous_Award = st.selectbox(
            "Previous Award", [0, 1],
            help="1 = Has received award before"
        )

        Training_score_average = st.slider(
            "Training Score Average", 0.0, 100.0, 50.0,
            help="Average score across trainings"
        )

        No_of_previous_employers = st.number_input(
            "Number of Previous Employers", 0, step=1,
            help="Total number of past employers"
        )

        submitted = st.form_submit_button("Predict Promotion")

    if submitted:
        input_df = pd.DataFrame([{
            "Trainings_Attended": Trainings_Attended,
            "Year_of_birth": Year_of_birth,
            "Last_performance_score": Last_performance_score,
            "Year_of_recruitment": Year_of_recruitment,
            "Targets_met": Targets_met,
            "Previous_Award": Previous_Award,
            "Training_score_average": Training_score_average,
            "No_of_previous_employers": No_of_previous_employers
        }])

        preds, probs = predict_promotion(input_df, threshold)

        st.markdown("### Prediction Result")
        st.write(f"Promotion Probability: **{probs[0]:.2f}**")

        if preds[0] == 1:
            st.success("Promotion Recommended (1)")
        else:
            st.error("Not Recommended for Promotion (0)")

        
        # SHAP EXPLANATION (HR ONLY)
        
        if is_hr:
            st.markdown("### Explanation (Why this decision?)")

            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)

            fig, ax = plt.subplots()
            shap.waterfall_plot(shap_values[0], show=False)
            st.pyplot(fig)


# AUTHOR INFORMATION
# -----------------------------------
st.divider()
st.markdown(
    """
    **Model Developer**  
    Irene Ufuoma Ayakazi  
    Data Analyst | AI & Machine Learning Engineer | Data Scientist  

    *This application supports HR decision-making and should not be used as the sole basis for promotion decisions.*
    """
)


