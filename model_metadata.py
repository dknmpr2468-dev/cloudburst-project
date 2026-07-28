import joblib

MODEL_PATH = "cloudburst_model.pkl"

data = joblib.load(MODEL_PATH)

print("Total Features :", len(data["feature_names"]))
print()

for i, f in enumerate(data["feature_names"], start=1):
    print(f"{i:02d}. {f}")