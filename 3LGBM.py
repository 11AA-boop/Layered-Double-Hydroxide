import os, importlib.machinery, importlib.util, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "22221...newdata.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RANDOM_STATE = 42
TEST_SIZE = 0.2
PDP_FEATURE_SPECS = [
    {"aliases": ["ts"], "label": "ts", "file_stem": "ts"},
    {"aliases": ["ta"], "label": "ta", "file_stem": "ta"},
    {"aliases": ["pH", "PH"], "label": "pH", "file_stem": "ph"},
    {
        "aliases": ["Tcal", "Tcal", "Tcal"],
        "label": "Tcal",
        "file_stem": "Tcal",
    },
    {"aliases": ["Co"], "label": "Co", "file_stem": "Co"},
    {"aliases": ["Ca"], "label": "Ca", "file_stem": "Ca"},
]

def _load_helper():
    candidates = ["1.py","1CatBoost.py"]
    for fname in os.listdir(BASE_DIR):
        if fname.startswith("1") and fname.endswith(".py") and fname not in candidates:
            candidates.append(fname)
    for fname in candidates:
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            loader = importlib.machinery.SourceFileLoader("helper_module", path)
            spec = importlib.util.spec_from_loader(loader.name, loader)
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            return module
    raise SystemExit("No helper file starting with '1' found.")

helper = _load_helper()

def build_pipeline(numeric_cols, binary_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("bin", binary_pipe, binary_cols),
    ], remainder="drop")
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        objective="regression",
        n_jobs=8,
        verbosity=-1,
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])

def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return r2, rmse, mae, mape

def plot_residuals(y_true, y_pred, output_dir):
    residuals = y_true - y_pred
    plt.figure(figsize=(7,5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (True - Pred)")
    plt.title("Residuals vs Predicted (LGBM)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_scatter_lgbm.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution (LGBM)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_hist_lgbm.png"), dpi=150)
    plt.close()

def plot_parity_with_marginals(y_train, pred_train, y_test, pred_test, r2_train, r2_test, output_dir):
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(8.4, 8.0))
    gs = gridspec.GridSpec(2,2,width_ratios=[4,1.2],height_ratios=[1.2,4],hspace=0.05,wspace=0.05)
    ax_scatter = plt.subplot(gs[1,0])
    ax_top = plt.subplot(gs[0,0], sharex=ax_scatter)
    ax_right = plt.subplot(gs[1,1], sharey=ax_scatter)

    ax_scatter.scatter(y_train, pred_train, s=24, alpha=0.7, color="#d95f02", label="Train")
    ax_scatter.scatter(y_test, pred_test, s=24, alpha=0.7, color="#1b9e77", label="Test")
    min_lim = min(y_train.min(), y_test.min(), pred_train.min(), pred_test.min())
    max_lim = max(y_train.max(), y_test.max(), pred_train.max(), pred_test.max())
    pad = 0.05 * (max_lim - min_lim)
    ax_scatter.plot([min_lim, max_lim],[min_lim, max_lim],"k--",linewidth=1.2)
    ax_scatter.set_xlim(min_lim - pad, max_lim + pad)
    ax_scatter.set_ylim(min_lim - pad, max_lim + pad)
    ax_scatter.set_xlabel("Experimental (qc)", fontsize=20, labelpad=9)
    ax_scatter.set_ylabel("Predicted (qc)", fontsize=20, labelpad=9)
    ax_scatter.tick_params(axis="both", labelsize=15)
    ax_scatter.legend(loc="upper left", fontsize=18, frameon=False)
    ax_scatter.text(
        0.91, 0.07,
        f"Train R$^2$: {r2_train:.3f}\nTest  R$^2$: {r2_test:.3f}",
        transform=ax_scatter.transAxes,
        fontsize=18,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.7, linewidth=0),
    )

    def _kde_plot(data, ax, color, orient="x"):
        try:
            kde = gaussian_kde(data)
            grid = np.linspace(min_lim - pad, max_lim + pad, 200)
            dens = kde(grid)
            if orient == "x":
                ax.plot(grid, dens, color=color, alpha=0.8)
                ax.fill_between(grid, dens, color=color, alpha=0.35)
            else:
                ax.plot(dens, grid, color=color, alpha=0.8)
                ax.fill_betweenx(grid, dens, color=color, alpha=0.35)
        except Exception:
            if orient == "x":
                ax.hist(data, bins=30, density=True, color=color, alpha=0.4)
            else:
                ax.hist(data, bins=30, density=True, color=color, alpha=0.4, orientation="horizontal")

    _kde_plot(y_train, ax_top, "#d95f02", "x")
    _kde_plot(y_test, ax_top, "#1b9e77", "x")
    ax_top.axis("off")
    _kde_plot(pred_train, ax_right, "#d95f02", "y")
    _kde_plot(pred_test, ax_right, "#1b9e77", "y")
    ax_right.axis("off")
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.14, top=0.97)
    fig.savefig(os.path.join(output_dir, "parity_joint_lgbm.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

def _find_col_case_insensitive(columns, names):
    targets = {name.lower() for name in names}
    for col in columns:
        if str(col).lower() in targets:
            return col
    return ""

def resolve_pdp_features(columns):
    resolved = []
    missing = []
    for spec in PDP_FEATURE_SPECS:
        col = _find_col_case_insensitive(columns, spec["aliases"])
        if col:
            resolved.append(
                {
                    "column": col,
                    "label": spec["label"],
                    "file_stem": spec["file_stem"],
                }
            )
        else:
            missing.append(spec["label"])
    if missing:
        print(f"[PDP] Missing features skipped: {', '.join(missing)}")
    return resolved

def extract_partial_dependence_curve(pd_result):
    grid_values = getattr(pd_result, "grid_values", None)
    if grid_values is None:
        grid_values = pd_result.get("grid_values")
    if grid_values is None:
        grid_values = getattr(pd_result, "values", None)
    if grid_values is None:
        grid_values = pd_result.get("values")

    average = getattr(pd_result, "average", None)
    if average is None:
        average = pd_result.get("average")

    x = np.asarray(grid_values[0], dtype=float).reshape(-1)
    y = np.asarray(average[0], dtype=float).reshape(-1)
    order = np.argsort(x)
    return x[order], y[order]

def smooth_pdp_curve(x_raw, y_raw):
    x_unique, unique_idx = np.unique(x_raw, return_index=True)
    y_unique = y_raw[unique_idx]
    if len(x_unique) <= 3:
        return x_unique, y_unique

    if len(y_unique) >= 7:
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    else:
        kernel = np.array([1, 2, 1], dtype=float)

    kernel = kernel / kernel.sum()
    pad = len(kernel) // 2
    y_padded = np.pad(y_unique, (pad, pad), mode="edge")
    y_trend = np.convolve(y_padded, kernel, mode="valid")
    x_smooth = np.linspace(x_unique.min(), x_unique.max(), 160)
    try:
        from scipy.interpolate import PchipInterpolator

        y_smooth = PchipInterpolator(x_unique, y_trend)(x_smooth)
    except Exception:
        y_smooth = np.interp(x_smooth, x_unique, y_trend)
    return x_smooth, y_smooth

def compute_pdp_curves(pipeline, X_pdp, pdp_features):
    from sklearn.inspection import partial_dependence

    curves = []
    for spec in pdp_features:
        pd_result = partial_dependence(
            pipeline,
            X_pdp,
            [spec["column"]],
            kind="average",
            grid_resolution=12,
        )
        x_raw, y_raw = extract_partial_dependence_curve(pd_result)
        x_smooth, y_smooth = smooth_pdp_curve(x_raw, y_raw)
        curves.append(
            {
                "spec": spec,
                "x_raw": x_raw,
                "y_raw": y_raw,
                "x_smooth": x_smooth,
                "y_smooth": y_smooth,
            }
        )
    return curves

def draw_pdp_panel(ax, curve, gray_color, blue_color, letter=None, show_legend=False):
    from matplotlib.lines import Line2D

    spec = curve["spec"]
    ax.plot(
        curve["x_raw"],
        curve["y_raw"],
        color=gray_color,
        linewidth=1.8,
        alpha=0.95,
        solid_capstyle="round",
        zorder=1,
    )
    ax.plot(
        curve["x_smooth"],
        curve["y_smooth"],
        color=blue_color,
        linewidth=2.0,
        alpha=0.95,
        solid_capstyle="round",
        zorder=2,
    )
    ax.set_xlabel(spec["label"])
    ax.set_ylabel("PDP")

    if letter is not None:
        ax.text(
            -0.12,
            1.03,
            letter,
            transform=ax.transAxes,
            fontsize=20,
            fontfamily="serif",
            va="top",
        )

    for spine in ax.spines.values():
        spine.set_color("#9f9f9f")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", colors="#444444", labelsize=9)
    ax.grid(False)

    if show_legend:
        ax.legend(
            handles=[
                Line2D([0], [0], color=gray_color, lw=1.8, label="ML prediction"),
                Line2D([0], [0], color=blue_color, lw=2.0, label="Fitted curve"),
            ],
            loc="upper right",
            frameon=False,
        )

def save_single_pdp_figures(curves, output_dir, gray_color, blue_color):
    for curve in curves:
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        draw_pdp_panel(
            ax,
            curve,
            gray_color=gray_color,
            blue_color=blue_color,
            show_legend=True,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"pdp_lgbm_{curve['spec']['file_stem']}.png"),
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

def plot_pdp(pipeline, X_train, pdp_features, output_dir):
    if not pdp_features:
        print("[PDP] No requested features found; skipping PDP plot.")
        return

    X_pdp = X_train.copy()
    for spec in pdp_features:
        X_pdp[spec["column"]] = pd.to_numeric(
            X_pdp[spec["column"]], errors="coerce"
        ).astype(float)

    gray_color = "#c9c9c9"
    blue_color = "#2f80c9"
    curves = compute_pdp_curves(pipeline, X_pdp, pdp_features)
    save_single_pdp_figures(curves, output_dir, gray_color, blue_color)

def plot_shap(model, preprocess, X_sample, feature_names, output_dir):
    try:
        X_trans = preprocess.transform(X_sample)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X_trans = np.asarray(X_trans, dtype=float)
        import shap
        warnings.filterwarnings("ignore", category=UserWarning)
        explainer = shap.Explainer(model.predict, X_trans)
        shap_values = explainer(X_trans)
        plt.figure()
        shap.summary_plot(shap_values.values, X_trans, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_lgbm.png"), dpi=150)
        plt.close()
        plt.figure()
        shap.summary_plot(shap_values.values, X_trans, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_bar_lgbm.png"), dpi=150)
        plt.close()
    except Exception as exc:
        print(f"[SHAP] skip due to error: {exc}")

def stability_cv(pipeline, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
    cv_rmse = -cross_val_score(pipeline, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    return float(cv_r2.mean()), float(cv_r2.std()), float(cv_rmse.mean()), float(cv_rmse.std())

def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"Data file not found: {DATA_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_raw = pd.read_excel(DATA_PATH)
    (
        df,
        target_col,
        feature_cols,
        binary_cols,
        group1_cols,
        group2_cols,
        temp_col,
    ) = helper._prepare_features(df_raw)

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    X = df[feature_cols]
    y = df[target_col]
    numeric_cols = [c for c in feature_cols if c not in binary_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    pipeline = build_pipeline(numeric_cols, binary_cols)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    r2, rmse, mae, mape = evaluate(y_test, y_pred)
    print("== Test Metrics (LGBM) ==")
    print(f"R2:   {r2:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"MAPE: {mape:.6f}%")

    plot_residuals(y_test, y_pred, OUTPUT_DIR)
    plot_parity_with_marginals(
        y_train.to_numpy(), y_pred_train, y_test.to_numpy(), y_pred, r2_train, r2, OUTPUT_DIR
    )

    cv_r2_mean, cv_r2_std, cv_rmse_mean, cv_rmse_std = stability_cv(pipeline, X, y)
    print("== Stability (5-fold CV) ==")
    print(f"R2 mean: {cv_r2_mean:.6f}, std: {cv_r2_std:.6f}")
    print(f"RMSE mean: {cv_rmse_mean:.6f}, std: {cv_rmse_std:.6f}")

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = helper._feature_names_from_preprocess(preprocess, numeric_cols, binary_cols)

    pdp_features = resolve_pdp_features(list(X_train.columns))
    plot_pdp(pipeline, X_train, pdp_features, OUTPUT_DIR)

    sample_n = min(500, len(X_train))
    X_sample = X_train.sample(n=sample_n, random_state=RANDOM_STATE)
    plot_shap(model, preprocess, X_sample, feature_names, OUTPUT_DIR)

    metrics_path = os.path.join(OUTPUT_DIR, "metrics_lgbm.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("== Test Metrics (LGBM) ==\n")
        f.write(f"R2:   {r2:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE:  {mae:.6f}\n")
        f.write(f"MAPE: {mape:.6f}%\n\n")
        f.write("== Stability (5-fold CV) ==\n")
        f.write(f"R2 mean: {cv_r2_mean:.6f}, std: {cv_r2_std:.6f}\n")
        f.write(f"RMSE mean: {cv_rmse_mean:.6f}, std: {cv_rmse_std:.6f}\n")
        f.write(
            "Top PDP features: "
            + ", ".join([item["label"] for item in pdp_features])
            + "\n"
        )

if __name__ == "__main__":
    main()
