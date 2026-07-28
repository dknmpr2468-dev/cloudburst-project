import joblib

# Load model file
saved = joblib.load("cloudburst_model.pkl")

# Extract objects
model = saved["model"]
scaler = saved["scaler"]
feature_names = saved["feature_names"]

print(model)
print(feature_names)