import joblib
import pandas as pd

# Load the saved model
model = joblib.load('xgb_promotion_model.pkl')

# Define the features used during training
features = [
    'Trainings_Attended',
    'Year_of_birth',
    'Last_performance_score',
    'Year_of_recruitment',
    'Targets_met',
    'Previous_Award',
    'Training_score_average',
    'No_of_previous_employers'
]

# Define the prediction function
def predict_promotion(data, threshold=0.50):
    # Ensure the input data contains the required features
    X = data[features]
    
    # Get predicted probabilities
    probs = model.predict_proba(X)[:, 1]
    
    # Apply threshold to get binary predictions
    preds = (probs >= threshold).astype(int)
    
    return preds, probs