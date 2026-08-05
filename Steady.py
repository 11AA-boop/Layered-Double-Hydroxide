import importlib.util
import os
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "22221...newdata.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUT_FIG_PNG = os.path.join(OUTPUT_DIR, "stability_r2_rmse_6models.png")
OUT_FIG_PDF = os.path.join(OUTPUT_DIR, "stability_r2_rmse_6models.pdf")
OUT_FIG_SVG = os.path.join(OUTPUT_DIR, "stability_r2_rmse_6models.svg")
OUT_CSV = os.path.join(OUTPUT_DIR, "stability_r2_rmse_6models.csv")
OUT_CAPTION = os.path.join(OUTPUT_DIR, "stability_r2_rmse_6models_caption.txt")

RANDOM_SEEDS = [32, 42, 52, 62, 72]
N_SPLITS = 5

MODEL_ORDER = ["CatBoost", "XGBoost", "LGBM", "RFR", "GBR", "BRT"]
COLORS = ["#df8e8e", "#90b7de", "#b8d9a4", "#8ecdc7", "#d5b3ea", "#f4c38c"]
FONT_FAMILY = ["Times New Roman", "Arial", "DejaVu Sans"]
BASE_FONT_SIZE = 20
CAPTION_TEXT = (
    "Figure X. Stability evaluation results of six machine learning regression "
    "models. Based on five-fold cross-validation across five predefined random "
    "seeds, violin "
    "plots show the distributions of training/testing RMSE and R^2 on the "
    "training and test sets. Black dots denote means and black horizontal "
    "lines denote medians. The evaluated models are CatBoost, XGBoost, "
    "LightGBM (LGBM), random forest regression (RFR), gradient boosting "
    "regression (GBR), and boosted regression trees (BRT)."
)
METRIC_STYLES = {
    "train_rmse": {
        "title": "(a) Training RMSE Stability",
        "ylabel": "Train RMSE",
    },
    "test_rmse": {
        "title": "(b) Testing RMSE Stability",
        "ylabel": "Test RMSE",
    },
    "train_r2": {
        "title": r"(c) Training R$^2$ Stability",
        "ylabel": "Train R$^2$",
    },
    "test_r2": {
        "title": r"(d) Testing R$^2$ Stability",
        "ylabel": "Test R$^2$",
    },
}

matplotlib.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": 30,
        "axes.labelsize": 26,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
)


def load_module(alias, filename):
    path = os.path.join(BASE_DIR, filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_data(cat_module):
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"Data file not found: {DATA_PATH}")

    df_raw = pd.read_excel(DATA_PATH)
    (
        df,
        target_col,
        feature_cols,
        binary_cols,
        _group1_cols,
        _group2_cols,
        _temp_col,
    ) = cat_module._prepare_features(df_raw)

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    X = df[feature_cols]
    y = df[target_col]
    numeric_cols = [c for c in feature_cols if c not in binary_cols]
    return X, y, numeric_cols, binary_cols


def sanitize_model_runtime(model):
    if not hasattr(model, "get_params"):
        return

    params = model.get_params(deep=False)
    updates = {}
    class_name = model.__class__.__name__

    if "n_jobs" in params:
        updates["n_jobs"] = 1
    if class_name == "LGBMRegressor" and "verbose" in params:
        updates["verbose"] = -1
    if class_name == "LGBMRegressor" and "verbosity" in params:
        updates["verbosity"] = -1
    if class_name == "XGBRegressor" and "verbosity" in params:
        updates["verbosity"] = 0

    if updates:
        model.set_params(**updates)


def evaluate_stability(X, y, builders):
    records = []
    for seed in RANDOM_SEEDS:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for model_name in MODEL_ORDER:
            print(f"[Run] {model_name} | seed={seed}")
            builder = builders[model_name]
            for fold_id, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]

                pipeline = builder(seed)
                sanitize_model_runtime(pipeline.named_steps["model"])
                pipeline.fit(X_train, y_train)

                pred_train = pipeline.predict(X_train)
                pred_test = pipeline.predict(X_test)

                train_rmse = float(np.sqrt(mean_squared_error(y_train, pred_train)))
                test_rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
                train_r2 = float(r2_score(y_train, pred_train))
                test_r2 = float(r2_score(y_test, pred_test))

                records.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold_id": fold_id,
                        "train_rmse": train_rmse,
                        "test_rmse": test_rmse,
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                    }
                )
    return pd.DataFrame(records)


def _draw_violin(ax, df, metric_col):
    meta = METRIC_STYLES[metric_col]
    series_list = [
        df.loc[df["model"] == name, metric_col].to_numpy(dtype=float)
        for name in MODEL_ORDER
    ]
    positions = np.arange(1, len(MODEL_ORDER) + 1)

    parts = ax.violinplot(
        series_list,
        positions=positions,
        widths=0.8,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[i])
        body.set_edgecolor("#555555")
        body.set_linewidth(0.8)
        body.set_alpha(0.82)

    means = [float(np.mean(values)) for values in series_list]
    medians = [float(np.median(values)) for values in series_list]

    for xpos, median in zip(positions, medians):
        ax.hlines(
            median,
            xpos - 0.18,
            xpos + 0.18,
            colors="#111111",
            linewidth=1.4,
            zorder=3,
        )

    ax.scatter(
        positions,
        means,
        s=22,
        c="#111111",
        zorder=4,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        MODEL_ORDER,
        rotation=32,
        ha="right",
        rotation_mode="anchor",
        fontsize=22,
        fontweight="bold",
    )
    ax.set_title(meta["title"], loc="left", pad=20, fontsize=30, fontweight="bold")
    ax.set_ylabel(meta["ylabel"], fontsize=26, fontweight="bold", labelpad=4)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.08, linestyle="-", linewidth=0.5, color="#8A8A8A")
    ax.tick_params(direction="out", length=4, colors="#333333", labelsize=22)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


def plot_stability(df):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16.2))
    _draw_violin(axes[0, 0], df, "train_rmse")
    _draw_violin(axes[0, 1], df, "test_rmse")
    _draw_violin(axes[1, 0], df, "train_r2")
    _draw_violin(axes[1, 1], df, "test_r2")

    fig.tight_layout(rect=(0.16, 0.08, 0.99, 0.94), pad=3.2, h_pad=3.0, w_pad=5.0)
    fig.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(OUT_FIG_SVG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def write_caption_file():
    with open(OUT_CAPTION, "w", encoding="utf-8-sig") as handle:
        handle.write(CAPTION_TEXT + "\n")


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cat_mod = load_module("m_cat", "1CatBoost.py")
    xgb_mod = load_module("m_xgb", "2XGBoost.py")
    lgbm_mod = load_module("m_lgbm", "3LGBM.py")
    rfr_mod = load_module("m_rfr", "4RFR.py")
    gbr_mod = load_module("m_gbr", "5GBR.py")
    brt_mod = load_module("m_brt", "6BRT.py")

    X, y, numeric_cols, binary_cols = prepare_data(cat_mod)

    builders = {
        "CatBoost": lambda seed: cat_mod._build_pipeline(
            numeric_cols, binary_cols, random_state=seed
        ),
        "XGBoost": lambda seed: xgb_mod.build_pipeline(
            numeric_cols, binary_cols, random_state=seed
        ),
        "LGBM": lambda seed: lgbm_mod.build_pipeline(
            numeric_cols, binary_cols, random_state=seed
        ),
        "RFR": lambda seed: rfr_mod.build_pipeline(
            numeric_cols, binary_cols, random_state=seed
        ),
        "GBR": lambda seed: gbr_mod.build_pipeline(
            numeric_cols, binary_cols, random_state=seed
        ),
        "BRT": lambda seed: brt_mod.build_pipeline(
            numeric_cols, binary_cols, random_state=seed
        ),
    }

    result_df = evaluate_stability(X, y, builders)
    result_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    plot_stability(result_df)
    write_caption_file()

    summary = (
        result_df.groupby("model")[["train_rmse", "test_rmse", "train_r2", "test_r2"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print("\n== Stability Summary (mean +/- std) ==")
    print(summary)
    print(f"\nSaved figure (PNG): {OUT_FIG_PNG}")
    print(f"Saved figure (PDF): {OUT_FIG_PDF}")
    print(f"Saved figure (SVG): {OUT_FIG_SVG}")
    print(f"Saved metrics: {OUT_CSV}")
    print(f"Saved caption: {OUT_CAPTION}")


if __name__ == "__main__":
    main()
