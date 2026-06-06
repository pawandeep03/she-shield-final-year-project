"""
SHESHIELD — Women Safety Risk Prediction System

This script:
  1) Loads your Jamshedpur dataset (CSV)
  2) Optionally "extends" it to a larger dataset by sampling + adding noise
  3) Trains a classifier (Decision Tree vs Logistic Regression)
  4) Saves model artifacts to: backend/model/model.pkl
  5) (Optional) Generates charts + a Folium safety map (backend/templates/map.html)

Run (recommended):
  python model_training.py --generate-assets --target-rows 5000
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


RISK_LABELS = ["Low", "Medium", "High"]


def _project_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent
    return {
        "root": root,
        "data_dir": root / "data",
        "model_dir": root / "model",
        "static_images": root / "static" / "images",
        "templates": root / "templates",
    }


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Basic cleanups to avoid accidental duplicates due to whitespace/case
    text_cols = ["Area_Name", "Lighting", "Time", "Crowd_Density", "Risk_Level"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in ["Lighting", "Time", "Crowd_Density", "Risk_Level"]:
        if col in df.columns:
            df[col] = df[col].str.title()

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def extend_dataset(df: pd.DataFrame, target_rows: int, seed: int = 42) -> pd.DataFrame:
    """
    "Large dataset" extension:
    - sample existing rows with replacement
    - add small noise to numeric columns to create realistic variation
    """
    if target_rows <= len(df):
        return df.copy()

    rng = np.random.default_rng(seed)
    n = int(target_rows)
    sampled = df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)

    # Add noise to numeric columns (keep them within sensible bounds)
    if "Crime_Rate" in sampled.columns:
        sampled["Crime_Rate"] = np.clip(
            sampled["Crime_Rate"].astype(float) + rng.normal(0, 1.0, size=n), 1, 10
        ).round(0).astype(int)
    if "Police_Distance" in sampled.columns:
        sampled["Police_Distance"] = np.clip(
            sampled["Police_Distance"].astype(float) + rng.normal(0, 0.35, size=n), 0.1, 10
        ).round(2)
    if "Latitude" in sampled.columns:
        sampled["Latitude"] = (sampled["Latitude"].astype(float) + rng.normal(0, 0.0015, size=n)).round(6)
    if "Longitude" in sampled.columns:
        sampled["Longitude"] = (sampled["Longitude"].astype(float) + rng.normal(0, 0.0015, size=n)).round(6)

    # Slightly perturb some categorical values to avoid perfect duplicates
    def _flip_with_prob(series: pd.Series, choices: list[str], p: float) -> pd.Series:
        mask = rng.random(len(series)) < p
        if mask.any():
            series = series.copy()
            series.loc[mask] = rng.choice(choices, size=int(mask.sum()))
        return series

    if "Lighting" in sampled.columns:
        sampled["Lighting"] = _flip_with_prob(sampled["Lighting"], ["Good", "Poor"], p=0.05)
    if "Time" in sampled.columns:
        sampled["Time"] = _flip_with_prob(sampled["Time"], ["Day", "Night"], p=0.05)
    if "Crowd_Density" in sampled.columns:
        sampled["Crowd_Density"] = _flip_with_prob(sampled["Crowd_Density"], ["Low", "Medium", "High"], p=0.05)

    # Keep Risk_Level from the sampled record (still "based on your data")
    # NOTE: For production you would re-label based on ground-truth sources.
    return sampled


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    lighting_map = {"Poor": 0, "Good": 1}
    time_map = {"Night": 0, "Day": 1}
    crowd_map = {"Low": 0, "Medium": 1, "High": 2}
    risk_map = {"Low": 0, "Medium": 1, "High": 2}

    df = df.copy()
    df["Lighting_Enc"] = df["Lighting"].map(lighting_map)
    df["Time_Enc"] = df["Time"].map(time_map)
    df["Crowd_Enc"] = df["Crowd_Density"].map(crowd_map)
    df["Risk_Encoded"] = df["Risk_Level"].map(risk_map)

    if df[["Lighting_Enc", "Time_Enc", "Crowd_Enc", "Risk_Encoded"]].isnull().any().any():
        bad = df[df[["Lighting_Enc", "Time_Enc", "Crowd_Enc", "Risk_Encoded"]].isnull().any(axis=1)]
        raise ValueError(
            "Found unexpected categorical values after encoding. "
            "Please check these rows:\n" + bad.head(10).to_string(index=False)
        )

    feature_cols = ["Crime_Rate", "Lighting_Enc", "Police_Distance", "Time_Enc", "Crowd_Enc"]
    X = df[feature_cols].astype(float)
    y = df["Risk_Encoded"].astype(int)

    meta = {
        "feature_cols": feature_cols,
        "class_labels": RISK_LABELS,
        "mappings": {
            "lighting": lighting_map,
            "time": time_map,
            "crowd": crowd_map,
            "risk": risk_map,
        },
    }
    return X, y, meta


def train_models(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    dt = DecisionTreeClassifier(max_depth=6, random_state=seed)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)

    lr = LogisticRegression(max_iter=1500, random_state=seed)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)

    if dt_acc >= lr_acc:
        best = {"model": dt, "model_name": "Decision Tree", "pred": dt_pred, "acc": float(dt_acc)}
    else:
        best = {"model": lr, "model_name": "Logistic Regression", "pred": lr_pred, "acc": float(lr_acc)}

    return {
        "best": best,
        "dt": {"acc": float(dt_acc)},
        "lr": {"acc": float(lr_acc)},
        "eval": {
            "X_test": X_test,
            "y_test": y_test,
            "confusion_matrix": confusion_matrix(y_test, best["pred"]),
            "classification_report": classification_report(
                y_test, best["pred"], target_names=RISK_LABELS, output_dict=False
            ),
        },
    }


def save_model(model_path: Path, model, meta: dict, accuracy: float, model_name: str) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "model_name": model_name,
        "accuracy": accuracy,
        **meta,
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)


def generate_assets(df: pd.DataFrame, train_out: dict, paths: dict[str, Path]) -> None:
    """
    Creates charts in static/images and a folium map in templates/map.html.
    """
    # Heavy libraries are imported only if needed
    import matplotlib

    matplotlib.use("Agg")  # safe for servers/headless environments
    import matplotlib.pyplot as plt
    import seaborn as sns

    paths["static_images"].mkdir(parents=True, exist_ok=True)
    paths["templates"].mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Chart 1: Risk distribution
    plt.figure(figsize=(8, 4))
    order = RISK_LABELS
    counts = df["Risk_Level"].value_counts().reindex(order).fillna(0)
    ax = sns.barplot(x=counts.index, y=counts.values)
    ax.set_title("Risk Level Distribution")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(int(v)), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(paths["static_images"] / "chart1_risk_distribution.png", dpi=150)
    plt.close()

    # Confusion matrix
    cm = train_out["eval"]["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=RISK_LABELS,
        yticklabels=RISK_LABELS,
        linewidths=0.5,
    )
    plt.title(f"Confusion Matrix — {train_out['best']['model_name']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(paths["static_images"] / "chart6_confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance (only for DT)
    best_model = train_out["best"]["model"]
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        names = ["Crime Rate", "Lighting", "Police Distance", "Time", "Crowd"]
        order_idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(9, 4))
        ax = sns.barplot(x=[names[i] for i in order_idx], y=importances[order_idx])
        ax.set_title("Feature Importance")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(paths["static_images"] / "chart7_feature_importance.png", dpi=150)
        plt.close()

    # Folium map
    try:
        import folium
    except Exception:
        # If folium isn't installed, still leave a helpful file
        (paths["templates"] / "map.html").write_text(
            "<h2>Map not available</h2><p>Install folium and re-run training with --generate-assets.</p>",
            encoding="utf-8",
        )
        return

    center_lat = float(df["Latitude"].mean())
    center_lon = float(df["Longitude"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

    color_map = {"Low": "green", "Medium": "orange", "High": "red"}
    # Use a smaller sample if df is huge, otherwise map becomes too heavy
    map_df = df.sample(n=min(len(df), 500), random_state=42).reset_index(drop=True)
    for _, r in map_df.iterrows():
        popup = (
            f"<b>{r['Area_Name']}</b><br>"
            f"Risk: {r['Risk_Level']}<br>"
            f"Crime Rate: {r['Crime_Rate']}<br>"
            f"Lighting: {r['Lighting']}<br>"
            f"Time: {r['Time']}<br>"
            f"Crowd: {r['Crowd_Density']}<br>"
            f"Police Distance: {r['Police_Distance']} km"
        )
        folium.CircleMarker(
            location=[float(r["Latitude"]), float(r["Longitude"])],
            radius=6,
            color=color_map.get(r["Risk_Level"], "blue"),
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup(popup, max_width=300),
        ).add_to(m)

    m.save(str(paths["templates"] / "map.html"))


def main() -> int:
    paths = _project_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(paths["data_dir"] / "data.csv"))
    parser.add_argument("--target-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-assets", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.input).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = load_and_clean(csv_path)
    df_big = extend_dataset(df, target_rows=args.target_rows, seed=args.seed)

    # Save extended dataset (useful for viva / report)
    (paths["data_dir"]).mkdir(parents=True, exist_ok=True)
    df_big.to_csv(paths["data_dir"] / "data_extended.csv", index=False)

    X, y, meta = encode_features(df_big)
    train_out = train_models(X, y, seed=args.seed)

    save_model(
        model_path=paths["model_dir"] / "model.pkl",
        model=train_out["best"]["model"],
        meta=meta,
        accuracy=train_out["best"]["acc"],
        model_name=train_out["best"]["model_name"],
    )

    if args.generate_assets:
        generate_assets(df_big, train_out, paths)

    print("✅ Training complete")
    print(f"   Rows used         : {len(df_big)}")
    print(f"   Best model        : {train_out['best']['model_name']}")
    print(f"   Accuracy (test)   : {train_out['best']['acc']*100:.2f}%")
    print(f"   Saved model       : {paths['model_dir'] / 'model.pkl'}")
    if args.generate_assets:
        print(f"   Saved map         : {paths['templates'] / 'map.html'}")
        print(f"   Saved charts      : {paths['static_images']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

