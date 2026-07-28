import json
import joblib
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = "D:\clou feature\cloudburst_model.pkl"

JSON_PATH = "point_31.6167_77.6167_2025-05-24.json"

# ==========================================================
# LOAD MODEL
# ==========================================================

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]

# ==========================================================
# LOAD JSON
# ==========================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    record = json.load(f)[0]

print("=" * 70)
print("JSON LOADED")
print("=" * 70)

# ==========================================================
# CHECK FEATURES
# ==========================================================

missing = []

values = []

for feature in feature_names:

    if feature not in record:
        missing.append(feature)
        values.append(0.0)
    else:
        values.append(record[feature])

if missing:

    print("\nMissing Features:\n")

    for m in missing:
        print(" -", m)

    print("\nPrediction cannot be trusted until these are generated.\n")

# ==========================================================
# DATAFRAME
# ==========================================================

X = pd.DataFrame([values], columns=feature_names)

# ==========================================================
# SCALE
# ==========================================================

X_scaled = scaler.transform(X)

# ==========================================================
# PREDICT
# ==========================================================

prediction = model.predict(X_scaled)[0]

probability = model.predict_proba(X_scaled)[0]

confidence = probability.max() * 100

# ==========================================================
# RESULT
# ==========================================================

print("\n")
print("=" * 70)
print(" CLOUD BURST PREDICTION")
print("=" * 70)

print(f"Prediction : {'CLOUDBURST' if prediction == 1 else 'NO CLOUDBURST'}")

print(f"Class      : {prediction}")

print("\nProbability")

print(f"No Cloudburst : {probability[0]:.6f}")

print(f"Cloudburst    : {probability[1]:.6f}")

print(f"\nConfidence : {confidence:.2f}%")

print("=" * 70)