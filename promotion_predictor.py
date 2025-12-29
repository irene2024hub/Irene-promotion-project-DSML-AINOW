
import pandas as pd
import xgboost as xgb

# Define the features used for prediction
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

# Load the XGBoost model from JSON
model = xgb.Booster()
model.load_model("xgb_promotion_model.json")

def predict_promotion(data: pd.DataFrame, threshold: float = 0.50):
    """
    Predicts promotion based on input data.

    Args:
        data (pd.DataFrame): Input data containing required features.
        threshold (float): Probability threshold for classification.

    Returns:
        preds (np.ndarray): Binary predictions (0 or 1).
        probs (np.ndarray): Predicted probabilities.
    """
    X = data[FEATURES]
    dmatrix = xgb.DMatrix(X)
    probs = model.predict(dmatrix)
    preds = (probs >= threshold).astype(int)
    return preds, probs