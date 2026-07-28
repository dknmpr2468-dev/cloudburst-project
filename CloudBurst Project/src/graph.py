import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# MODEL PATH
# ==========================================================
MODEL_PATH = r"D:\clou feature\cloudburst_model.pkl"

# ==========================================================
# LOAD MODEL
# ==========================================================
obj = joblib.load(MODEL_PATH)

# ==========================================================
# FIND MODEL INSIDE PKL
# ==========================================================
if hasattr(obj, "feature_importances_"):
    model = obj

elif isinstance(obj, dict):

    model = None

    for key, value in obj.items():

        if hasattr(value, "feature_importances_"):

            model = value
            print(f"Model found inside key : {key}")
            break

    if model is None:
        raise Exception("No ML model found inside the dictionary.")

else:
    raise Exception("Unsupported PKL format.")

# ==========================================================
# FEATURE NAMES
# ==========================================================
if hasattr(model, "feature_names_"):
    feature_names = model.feature_names_

elif hasattr(model, "feature_names"):
    feature_names = model.feature_names

else:
    feature_names = [f"Feature {i+1}" for i in range(len(model.feature_importances_))]

# ==========================================================
# CREATE DATAFRAME
# ==========================================================
df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

df = df.sort_values(
    by="Importance",
    ascending=False
)

# Save CSV
df.to_csv(
    r"D:\clou feature\feature_importance.csv",
    index=False
)

# ==========================================================
# TOP 20 FEATURES
# ==========================================================
top20 = df.head(20).copy()

# ==========================================================
# CLEAN FEATURE NAMES
# ==========================================================
top20["Feature"] = (
    top20["Feature"]
    .str.replace("_", " ", regex=False)
    .str.replace("2m", "(2 m)", regex=False)
    .str.replace("10m", "(10 m)", regex=False)
    .str.replace("100m", "(100 m)", regex=False)
    .str.replace("7d", "7-Day", regex=False)
    .str.replace("3d", "3-Day", regex=False)
    .str.title()
)

# ==========================================================
# GRAPH
# ==========================================================
plt.style.use("default")

fig, ax = plt.subplots(figsize=(13,9))

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.barh(
    top20["Feature"][::-1],
    top20["Importance"][::-1],
    color="#1565C0",
    edgecolor="black",
    linewidth=0.7
)

# ==========================================================
# TITLE
# ==========================================================
ax.set_title(
    "Top 20 Feature Importance (CatBoost)",
    fontsize=20,
    fontweight="bold",
    pad=20
)

ax.set_xlabel(
    "Importance Score",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel(
    "Features",
    fontsize=14,
    fontweight="bold"
)

# ==========================================================
# GRID
# ==========================================================
ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.30
)

# ==========================================================
# VALUE LABELS
# ==========================================================
for bar in bars:

    value = bar.get_width()

    ax.text(
        value + 0.10,
        bar.get_y() + bar.get_height()/2,
        f"{value:.2f}",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

# ==========================================================
# REMOVE EXTRA BORDERS
# ==========================================================
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.tight_layout()

# ==========================================================
# SAVE IMAGE
# ==========================================================
plt.savefig(
    r"D:\clou feature\Top20_Feature_Importance_Professional.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\nDone Successfully!")
print("Image Saved : D:\\clou feature\\Top20_Feature_Importance_Professional.png")