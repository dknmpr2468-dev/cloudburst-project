"""
Pan-India Cloudburst Prediction — Inference Script
====================================================
Load the saved best model and make predictions on new data.

Usage:
    python predict.py --input <path_to_csv_or_xlsx>
    python predict.py --lat 30.5 --lon 78.2 --features feature_file.csv
    python predict.py --demo
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "cloudburst_model.pkl")


def load_model(model_path=MODEL_PATH):
    """Load the saved model artifact."""
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        print("   Please run train.py first to train and save the model.")
        sys.exit(1)

    artifact = joblib.load(model_path)
    print(f"✅ Loaded model: {artifact['model_name']}")
    print(f"   Features expected: {len(artifact['feature_names'])}")
    return artifact


def preprocess_input(df, artifact):
    """
    Preprocess input data to match training format.
    Drops the same columns that were dropped during training,
    aligns feature columns, and applies the saved scaler.
    """
    # Drop columns that were dropped during training
    preprocessing_info = artifact["preprocessing"]
    feature_names = artifact["feature_names"]
    scaler = artifact["scaler"]

    # Drop leakage and metadata columns
    for col_list_key in ["dropped_columns", "dropped_non_numeric", "dropped_high_corr", "dropped_zero_var"]:
        cols = preprocessing_info.get(col_list_key, [])
        cols_present = [c for c in cols if c in df.columns]
        if cols_present:
            df = df.drop(columns=cols_present)

    # Drop Target if present
    if "Target" in df.columns:
        df = df.drop(columns=["Target"])

    # Drop remaining non-numeric columns
    non_numeric = df.select_dtypes(include=["object", "string", "datetime"]).columns.tolist()
    if non_numeric:
        df = df.drop(columns=non_numeric)

    # Align columns to match training features
    missing_features = [f for f in feature_names if f not in df.columns]
    extra_features = [f for f in df.columns if f not in feature_names]

    if missing_features:
        print(f"   ⚠️  Missing features (filled with 0): {missing_features[:10]}...")
        for f in missing_features:
            df[f] = 0

    if extra_features:
        print(f"   ⚠️  Extra features (dropped): {extra_features[:10]}...")
        df = df.drop(columns=extra_features)

    # Reorder columns to match training order
    df = df[feature_names]

    # Handle missing/infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())

    # Apply scaler
    X_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature_names,
        index=df.index
    )

    return X_scaled


def predict(X_scaled, artifact):
    """Run prediction using the loaded model."""
    model = artifact["model"]
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    return predictions, probabilities


def predict_from_file(input_path, model_path=MODEL_PATH, output_path=None):
    """
    Full prediction pipeline from file input.

    Parameters:
        input_path: Path to CSV or Excel file with weather features
        model_path: Path to the saved model .pkl file
        output_path: Optional path to save predictions

    Returns:
        DataFrame with predictions
    """
    print("\n" + "=" * 60)
    print("  🌩️  CLOUDBURST PREDICTION — INFERENCE")
    print("=" * 60)

    # Load model
    artifact = load_model(model_path)

    # Load input data
    print(f"\n📂 Loading input data: {input_path}")
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_path)
    else:
        print(f"❌ Unsupported file format. Use .csv or .xlsx")
        sys.exit(1)

    print(f"   Input shape: {df.shape}")

    # Store location info if available
    location_info = {}
    for col in ["Date", "Location", "Latitude", "Longitude"]:
        if col in df.columns:
            location_info[col] = df[col].copy()

    # Preprocess
    print("\n🔧 Preprocessing...")
    X_scaled = preprocess_input(df, artifact)
    print(f"   Preprocessed shape: {X_scaled.shape}")

    # Predict
    print("\n🔮 Running predictions...")
    predictions, probabilities = predict(X_scaled, artifact)

    # Build results DataFrame
    results = pd.DataFrame()
    for col_name, col_data in location_info.items():
        results[col_name] = col_data
    results["Prediction"] = predictions
    results["Probability"] = probabilities
    results["Risk_Level"] = pd.cut(
        probabilities,
        bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=["Low", "Moderate", "High", "Critical"]
    )
    results["Label"] = results["Prediction"].map({0: "No Cloudburst", 1: "Cloudburst"})

    # Display summary
    print("\n" + "=" * 60)
    print("  📊 PREDICTION SUMMARY")
    print("=" * 60)
    total = len(results)
    cb_count = int(predictions.sum())
    print(f"   Total samples:     {total}")
    print(f"   Cloudburst:        {cb_count} ({cb_count/total*100:.1f}%)")
    print(f"   No Cloudburst:     {total - cb_count} ({(total-cb_count)/total*100:.1f}%)")
    print(f"\n   Risk Distribution:")
    print(f"   {results['Risk_Level'].value_counts().to_string()}")

    print(f"\n   Average probability: {probabilities.mean():.4f}")
    print(f"   Max probability:     {probabilities.max():.4f}")
    print(f"   Min probability:     {probabilities.min():.4f}")

    # Show top 10 highest risk predictions
    print(f"\n   🔴 Top 10 Highest Risk Predictions:")
    top10 = results.nlargest(10, "Probability")
    display_cols = [c for c in ["Location", "Date", "Probability", "Risk_Level", "Label"] if c in top10.columns]
    if not display_cols:
        display_cols = ["Probability", "Risk_Level", "Label"]
    print(top10[display_cols].to_string(index=False))

    # Save results
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_predictions.csv"
    results.to_csv(output_path, index=False)
    print(f"\n   💾 Predictions saved to: {output_path}")

    return results


def demo_prediction(model_path=MODEL_PATH):
    """Run a demo prediction using the training data."""
    print("\n" + "=" * 60)
    print("  🌩️  CLOUDBURST PREDICTION — DEMO MODE")
    print("=" * 60)

    artifact = load_model(model_path)

    # Load original dataset for demo
    demo_path = os.path.join(SCRIPT_DIR, "cloudburst_dataset_with_features_20260715_195016.xlsx")
    if not os.path.exists(demo_path):
        print("❌ Demo dataset not found. Please provide an input file.")
        sys.exit(1)

    df = pd.read_excel(demo_path)
    # Take a small random sample
    sample = df.sample(n=min(20, len(df)), random_state=42)

    location_info = {}
    for col in ["Date", "Location", "Latitude", "Longitude", "Target"]:
        if col in sample.columns:
            location_info[col] = sample[col].values

    X_scaled = preprocess_input(sample.copy(), artifact)
    predictions, probabilities = predict(X_scaled, artifact)

    print("\n📊 Demo Predictions (20 random samples):")
    print("-" * 80)
    demo_results = pd.DataFrame()
    if "Location" in location_info:
        demo_results["Location"] = location_info["Location"]
    if "Target" in location_info:
        demo_results["Actual"] = location_info["Target"]
    demo_results["Predicted"] = predictions
    demo_results["Probability"] = [f"{p:.4f}" for p in probabilities]
    demo_results["Match"] = ["✅" if a == p else "❌"
                              for a, p in zip(location_info.get("Target", predictions), predictions)]
    print(demo_results.to_string(index=False))

    if "Target" in location_info:
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(location_info["Target"], predictions)
        print(f"\n   Demo Accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🌩️ Pan-India Cloudburst Prediction — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --demo
  python predict.py --input new_weather_data.csv
  python predict.py --input data.xlsx --output results.csv
  python predict.py --model custom_model.pkl --input data.csv
        """
    )
    parser.add_argument("--input", "-i", type=str, help="Path to input CSV/Excel file")
    parser.add_argument("--output", "-o", type=str, help="Path to save prediction results")
    parser.add_argument("--model", "-m", type=str, default=MODEL_PATH, help="Path to model .pkl file")
    parser.add_argument("--demo", action="store_true", help="Run demo prediction on sample data")

    args = parser.parse_args()

    if args.demo:
        demo_prediction(args.model)
    elif args.input:
        predict_from_file(args.input, args.model, args.output)
    else:
        parser.print_help()
        print("\n💡 Quick start: python predict.py --demo")
