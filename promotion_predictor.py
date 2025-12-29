import joblib
import pandas as pd

# Load model once
model = joblib.load("xgb_promotion_model.pkl")

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




