from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "visualizations"
OUT.mkdir(exist_ok=True)

def analyze(filename, label):
    df = pd.read_csv(DATA / filename)
    print(f"\\n{label} wine: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Missing cells:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    clean = df.drop_duplicates().copy()
    clean.to_csv(ROOT / f"cleaned_{label.lower()}_wine.csv", index=False)
    print("Rows after duplicate removal:", len(clean))
    print("Mean quality:", round(clean.quality.mean(), 3))
    X, y = clean.drop(columns="quality"), clean.quality
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("MAE:", round(mean_absolute_error(y_test, pred), 3))
    print("RMSE:", round(mean_squared_error(y_test, pred) ** .5, 3))
    print("R2:", round(r2_score(y_test, pred), 3))
    clean.groupby("quality").alcohol.mean().plot(kind="bar")
    plt.xlabel("Quality score"); plt.ylabel("Average alcohol")
    plt.title(f"{label} wine: Average alcohol by quality")
    plt.tight_layout(); plt.savefig(OUT / f"{label.lower()}_alcohol_by_quality.png", dpi=150); plt.close()

if __name__ == "__main__":
    analyze("winequality-red.csv", "Red")
    analyze("winequality-white.csv", "White")
