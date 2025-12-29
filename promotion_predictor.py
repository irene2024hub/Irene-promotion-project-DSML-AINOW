import joblib
import pandas as pd


import xgboost as xgb

# Load the XGBoost model from JSON
model = xgb.Booster()
model.load_model("xgb_promotion_model.json")

# Convert input DataFrame to DMatrix
dtest = xgb.DMatrix(input_df)  # input_df is your user-uploaded data

# Predict
preds = model.predict(dtest)

FEATURES = [
    "Trainings_Attended",
    "Year_of_birth",
    "Last_performance_score",
    "Year_of_recruitment",
    "Targets_met",
    "Previous_Award",
    "Training_score_average",
    "No_of_previous_employers"
]

def predict_promotion(data: pd.DataFrame, threshold: float = 0.50):
    """
    Returns:
    - predictions (0 or 1)
    - probabilities
    """
    X = data[FEATURES]
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds, probs




