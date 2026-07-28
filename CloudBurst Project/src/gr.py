import pickle
import joblib
import traceback
from catboost import CatBoostClassifier

model_path = r"D:\clou feature\cloudburst_model.pkl"

print("=" * 60)
print("CHECKING MODEL FILE")
print("=" * 60)

# -----------------------------
# Method 1 : Pickle
# -----------------------------
try:
    print("\nTrying pickle.load() ...")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("SUCCESS")
    print("Model Type:", type(model))

    if hasattr(model, "feature_names_"):
        print("\nTotal Features:", len(model.feature_names_))

        for i, feature in enumerate(model.feature_names_, 1):
            print(f"{i}. {feature}")

    else:
        print("\nfeature_names_ attribute not found.")
        print(dir(model))

except Exception as e:
    print("\nPickle Failed")
    traceback.print_exc()

# -----------------------------
# Method 2 : Joblib
# -----------------------------
try:
    print("\nTrying joblib.load() ...")

    model = joblib.load(model_path)

    print("SUCCESS")
    print("Model Type:", type(model))

    if hasattr(model, "feature_names_"):
        print("\nTotal Features:", len(model.feature_names_))

        for i, feature in enumerate(model.feature_names_, 1):
            print(f"{i}. {feature}")

    else:
        print("\nfeature_names_ attribute not found.")
        print(dir(model))

except Exception as e:
    print("\nJoblib Failed")
    traceback.print_exc()

# -----------------------------
# Method 3 : CatBoost Native
# -----------------------------
try:
    print("\nTrying CatBoost load_model() ...")

    model = CatBoostClassifier()
    model.load_model(model_path)

    print("SUCCESS")
    print("\nTotal Features:", len(model.feature_names_))

    for i, feature in enumerate(model.feature_names_, 1):
        print(f"{i}. {feature}")

except Exception as e:
    print("\nCatBoost Native Failed")
    traceback.print_exc()