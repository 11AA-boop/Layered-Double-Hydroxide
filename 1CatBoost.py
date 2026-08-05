import os
import re
import sys
import warnings
import numpy as np

def _require_package(name, extra_hint=""):
    try:
        return __import__(name)
    except Exception as exc:  
        msg = f"Missing package: {name}. {extra_hint}".strip()
        raise SystemExit(msg) from exc


pd = _require_package("pandas", "Install with: pip install pandas openpyxl")
sklearn = _require_package("sklearn", "Install with: pip install scikit-learn")
_ = _require_package("matplotlib", "Install with: pip install matplotlib")
catboost = _require_package("catboost", "Install with: pip install catboost")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from scipy.stats import gaussian_kde
try:
    import shap  
except Exception:  
    shap = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "22221...newdata.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
JOINT_PLOT_FILENAME = "joint_distribution_catboost.png"
LEGACY_JOINT_PLOT_FILENAME = "parity_joint_catboost.png"
PDP_PLOT_FILENAME = "pdp_selected_6features.png"
LEGACY_PDP_PLOT_FILENAME = "pdp.png"
RANDOM_SEEDS = [32, 42, 52, 62, 72]
REPRESENTATIVE_SEED = 42
# All 12 material-category columns.  These columns are one-hot/binary features
# and must never fall through to the numeric pipeline.
MATERIAL_BASES = [
    "Mg-Al",
    "Mg-Fe",
    "Cu-AL",
    "Zn-Al",
    "Ca-Al",
    "Ca-Fe",
    "Ca-Mg",
    "Zn-Fe",
    "Ni-Fe",
    "Fe-Al",
    "Mg-La",
    "Mn-Fe",
]

# The material columns are mutually exclusive (one-hot encoded).
GROUP1_BASES = MATERIAL_BASES

GROUP2_BASES = [
    "Zr-",
    "La-",
    "CO3-",
    "Cl-",
    "NO3-",
]

SPECIAL_NUM_BASES = ["ta", "Tads", "Co", "Ca"]
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

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 13,
        "figure.titlesize": 17,
    }
)

def _normalize_col(col):
    return str(col).strip()

def _base_name(col):
    return re.sub(r"\.\d+$", "", col)

def _find_col_case_insensitive(columns, names):
    targets = {n.lower() for n in names}
    for col in columns:
        if col.lower() in targets:
            return col
    return None

def _get_cols_by_bases(columns, bases):
    # Material headers are case-sensitive: Mg-Al and Mg-AL are distinct
    # fields and must never be merged during feature detection.
    bases_set = {str(base).strip() for base in bases}
    matched = []
    for col in columns:
        if _base_name(_normalize_col(col)) in bases_set:
            matched.append(col)
    return matched

def _coerce_binary(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = (df[col] != 0).astype(int)

def _coerce_numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def _enforce_exactly_one(df, group_cols, raw_values):
    if not group_cols:
        return {"fixed": 0, "total": 0}

    raw = raw_values.to_numpy()
    binary = (raw != 0).astype(int)
    sums = binary.sum(axis=1)
    invalid = sums != 1
    fixed = binary.copy()
    col_freq = binary.sum(axis=0)

    idxs = np.where(invalid)[0]
    for i in idxs:
        fixed[i, :] = 0
        row_raw = raw[i]
        if sums[i] == 0:
            if np.all(row_raw == 0):
                chosen = int(np.argmax(col_freq))
            else:
                chosen = int(np.argmax(row_raw))
        else:
            chosen = int(np.argmax(row_raw))
        fixed[i, chosen] = 1

    df[group_cols] = pd.DataFrame(fixed, columns=group_cols, index=df.index).astype(
        int
    )
    return {"fixed": int(invalid.sum()), "total": len(df)}

def _enforce_at_most_one(df, group_cols, raw_values):
    if not group_cols:
        return {"fixed": 0, "total": 0}

    raw = raw_values.to_numpy()
    binary = (raw != 0).astype(int)
    sums = binary.sum(axis=1)
    invalid = sums > 1
    fixed = binary.copy()
    col_freq = binary.sum(axis=0)

    idxs = np.where(invalid)[0]
    for i in idxs:
        fixed[i, :] = 0
        row_raw = raw[i]
        if np.all(row_raw == 0):
            chosen = int(np.argmax(col_freq))
        else:
            chosen = int(np.argmax(row_raw))
        fixed[i, chosen] = 1

    df[group_cols] = pd.DataFrame(fixed, columns=group_cols, index=df.index).astype(
        int
    )
    return {"fixed": int(invalid.sum()), "total": len(df)}


def _constraint_report(df, group1_cols, group2_cols):
    if group1_cols:
        group1_sum = df[group1_cols].sum(axis=1)
        ok_group1 = (group1_sum == 1).mean() * 100
        print(
            f"[Check] Group1(one-hot) OK rows: {ok_group1:.2f}% "
            f"(expected 100%)."
        )
    if group2_cols:
        group2_sum = df[group2_cols].sum(axis=1)
        ok_group2 = (group2_sum <= 1).mean() * 100
        print(
            f"[Check] Group2(<=1 hot) OK rows: {ok_group2:.2f}% "
            f"(expected 100%)."
        )

def _prepare_features(df):
    df = df.copy()
    df.columns = [_normalize_col(c) for c in df.columns]

    target_col = _find_col_case_insensitive(df.columns, ["qc"])
    if not target_col:
        raise SystemExit("Target column 'qc' not found.")

    group1_cols = _get_cols_by_bases(df.columns, GROUP1_BASES)
    group2_cols = _get_cols_by_bases(df.columns, GROUP2_BASES)

    # Fail loudly instead of silently treating a missing material column as a
    # continuous feature.  This keeps the model schema explicitly binary for
    # every declared material category.
    matched_material_bases = {_base_name(col) for col in group1_cols}
    missing_materials = [
        base
        for base in MATERIAL_BASES
        if base not in matched_material_bases
    ]
    if missing_materials:
        raise SystemExit(
            "Required material columns not found: "
            + ", ".join(missing_materials)
        )

    temp_col = _find_col_case_insensitive(df.columns, ["Tcal", "Tcal"])
    if temp_col and temp_col not in df.columns:
        temp_col = None

    binary_cols = sorted(set(group1_cols + group2_cols))
    feature_cols = [c for c in df.columns if c != target_col]

    raw_group1 = df[group1_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    raw_group2 = df[group2_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    if group1_cols:
        initial_ok = (raw_group1.ne(0).sum(axis=1) == 1).mean() * 100
        print(
            f"[Check] Group1(one-hot) OK rows before fix: {initial_ok:.2f}% "
            f"(expected 100%)."
        )
    if group2_cols:
        initial_ok = (raw_group2.ne(0).sum(axis=1) <= 1).mean() * 100
        print(
            f"[Check] Group2(<=1 hot) OK rows before fix: {initial_ok:.2f}% "
            f"(expected 100%)."
        )

    fix1 = _enforce_exactly_one(df, group1_cols, raw_group1)
    fix2 = _enforce_at_most_one(df, group2_cols, raw_group2)

    if fix1["fixed"]:
        print(
            f"[Fix] Group1 adjusted rows: {fix1['fixed']} / {fix1['total']}."
        )
    if fix2["fixed"]:
        print(
            f"[Fix] Group2 adjusted rows: {fix2['fixed']} / {fix2['total']}."
        )

    # Enforce integer 0/1 representation for every recognized binary field.
    _coerce_binary(df, binary_cols)
    _coerce_numeric(df, [c for c in feature_cols if c not in binary_cols])

    _constraint_report(df, group1_cols, group2_cols)

    return df, target_col, feature_cols, binary_cols, group1_cols, group2_cols, temp_col

def _build_pipeline(numeric_cols, binary_cols, random_state=REPRESENTATIVE_SEED):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("bin", binary_pipe, binary_cols),
        ],
        remainder="drop",
    )

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

def _evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    return r2, rmse, mae, mape


def _plot_residuals(y_true, y_pred, output_dir):
    residuals = y_true - y_pred

    x_min = float(np.min(y_pred))
    x_max = float(np.max(y_pred))
    x_pad = max((x_max - x_min) * 0.02, 1.0)
    band_x = np.array([x_min - x_pad, x_max + x_pad])

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.6, zorder=2)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlim(band_x[0], band_x[-1])
    plt.xlabel("Predicted")
    plt.ylabel("Residual (True - Pred)")
    plt.title("Residuals vs Predicted (CatBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_scatter.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_hist.png"), dpi=150)
    plt.close()

def _plot_parity_with_marginals(
    y_train,
    pred_train,
    y_test,
    pred_test,
    r2_train,
    r2_test,
    output_dir,
):
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[4.2, 1.15],
        height_ratios=[1.15, 4.2],
        hspace=0.05,
        wspace=0.08,
    )

    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    train_color = "#ee7f2d"
    test_color = "#39b29c"
    point_size = 30

    ax_scatter.scatter(
        y_train,
        pred_train,
        s=point_size,
        alpha=0.82,
        color=train_color,
        label="Train",
        edgecolors="none",
    )
    ax_scatter.scatter(
        y_test,
        pred_test,
        s=point_size,
        alpha=0.82,
        color=test_color,
        label="Test",
        edgecolors="none",
    )
    min_lim = min(y_train.min(), y_test.min(), pred_train.min(), pred_test.min())
    max_lim = max(y_train.max(), y_test.max(), pred_train.max(), pred_test.max())
    span = max_lim - min_lim
    pad = max(span * 0.05, 1.0)

    ax_scatter.plot(
        [min_lim - pad, max_lim + pad],
        [min_lim - pad, max_lim + pad],
        color="#1f1f1f",
        linestyle="--",
        linewidth=1.2,
    )
    ax_scatter.set_xlim(min_lim - pad, max_lim + pad)
    ax_scatter.set_ylim(min_lim - pad, max_lim + pad)
    ax_scatter.set_xlabel("Experimental (qc)")
    ax_scatter.set_ylabel("Predicted")
    ax_scatter.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#d0d0d0",
        framealpha=0.95,
    )
    ax_scatter.text(
        0.58,
        0.08,
        f"Train R$^2$: {r2_train:.3f}\nTest R$^2$: {r2_test:.3f}",
        transform=ax_scatter.transAxes,
        fontsize=15,
        verticalalignment="bottom",
        horizontalalignment="left",
    )

    for spine in ax_scatter.spines.values():
        spine.set_color("#4f4f4f")
        spine.set_linewidth(1.0)

    def _kde_plot(data, ax, color, orient="x"):
        try:
            kde = gaussian_kde(data)
            grid = np.linspace(min_lim - pad, max_lim + pad, 200)
            dens = kde(grid)
            if orient == "x":
                ax.plot(grid, dens, color=color, linewidth=1.6)
                ax.fill_between(grid, dens, color=color, alpha=0.26)
            else:
                ax.plot(dens, grid, color=color, linewidth=1.6)
                ax.fill_betweenx(grid, dens, color=color, alpha=0.26)
        except Exception:
            if orient == "x":
                ax.hist(data, bins=30, density=True, color=color, alpha=0.3)
            else:
                ax.hist(
                    data,
                    bins=30,
                    density=True,
                    color=color,
                    alpha=0.3,
                    orientation="horizontal",
                )

    _kde_plot(y_train, ax_top, train_color, "x")
    _kde_plot(y_test, ax_top, test_color, "x")
    ax_top.axis("off")
    _kde_plot(pred_train, ax_right, train_color, "y")
    _kde_plot(pred_test, ax_right, test_color, "y")
    ax_right.axis("off")
    fig.savefig(
        os.path.join(output_dir, JOINT_PLOT_FILENAME),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    legacy_plot = os.path.join(output_dir, LEGACY_JOINT_PLOT_FILENAME)
    if os.path.exists(legacy_plot):
        os.remove(legacy_plot)


def _feature_names_from_preprocess(preprocess, numeric_cols, binary_cols):
    try:
        names = preprocess.get_feature_names_out()
        return [n.replace("num__", "").replace("bin__", "") for n in names]
    except Exception:
        return list(numeric_cols) + list(binary_cols)


def _resolve_pdp_features(columns):
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


def _extract_partial_dependence_curve(pd_result):
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


def _smooth_pdp_curve(x_raw, y_raw):
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


def _compute_pdp_curves(pipeline, X_pdp, pdp_features):
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
        x_raw, y_raw = _extract_partial_dependence_curve(pd_result)
        x_smooth, y_smooth = _smooth_pdp_curve(x_raw, y_raw)
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

def _draw_pdp_panel(ax, curve, gray_color, blue_color, letter=None, show_legend=False):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    spec = curve["spec"]
    ax.plot(
        curve["x_raw"],
        curve["y_raw"],
        color=gray_color,
        linewidth=2.1,
        alpha=0.95,
        solid_capstyle="round",
        zorder=1,
    )
    ax.plot(
        curve["x_smooth"],
        curve["y_smooth"],
        color=blue_color,
        linewidth=2.4,
        alpha=0.95,
        solid_capstyle="round",
        zorder=2,
    )
    ax.set_xlabel(spec["label"], fontsize=16, fontweight="bold", labelpad=7)
    ax.set_ylabel("PDP", fontsize=16, fontweight="bold", labelpad=7)

    if spec["label"].casefold() == "ph":
        ax.set_xlim(left=3.0)
        x_upper = float(np.nanmax(curve["x_smooth"]))
        ax.set_xticks(np.arange(3.0, np.ceil(x_upper) + 1.0, 2.0))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if letter is not None:
        ax.text(
            -0.12,
            1.02,
            letter,
            transform=ax.transAxes,
            fontsize=26,
            fontweight="bold",
            fontfamily="serif",
            va="bottom",
        )

    for spine in ax.spines.values():
        spine.set_color("#9f9f9f")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", colors="#222222", labelsize=14, width=1.2)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.grid(False)

    if show_legend:
        ax.legend(
            handles=[
                Line2D([0], [0], color=gray_color, lw=2.1, label="ML prediction"),
                Line2D([0], [0], color=blue_color, lw=2.4, label="Fitted curve"),
            ],
            loc="upper right",
            frameon=False,
            prop={"size": 12, "weight": "bold"},
        )


def _save_single_pdp_figures(curves, output_dir, gray_color, blue_color):
    for curve in curves:
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        _draw_pdp_panel(
            ax,
            curve,
            gray_color=gray_color,
            blue_color=blue_color,
            show_legend=True,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"pdp_{curve['spec']['file_stem']}.png"),
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)


def _plot_feature_importance(model, feature_names, output_dir):
    importances = model.get_feature_importance()
    order = np.argsort(importances)[::-1]
    top_idx = order[: min(20, len(order))]
    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_names[i] for i in top_idx][::-1],
        importances[top_idx][::-1],
    )
    plt.xlabel("Importance")
    plt.title("Embedded Feature Importance (CatBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efi_feature_importance.png"), dpi=150)
    plt.close()


def _plot_pdp(pipeline, X_train, pdp_features, output_dir):
    from matplotlib.lines import Line2D

    if not pdp_features:
        print("[PDP] No requested features found; skipping PDP plot.")
        return

    X_pdp = X_train.copy()
    for spec in pdp_features:
        X_pdp[spec["column"]] = pd.to_numeric(
            X_pdp[spec["column"]], errors="coerce"
        ).astype(float)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    axes = axes.ravel()
    gray_color = "#c9c9c9"
    blue_color = "#46a3f7"
    curves = _compute_pdp_curves(pipeline, X_pdp, pdp_features)

    for idx, (ax, curve) in enumerate(zip(axes, curves)):
        _draw_pdp_panel(
            ax,
            curve,
            gray_color=gray_color,
            blue_color=blue_color,
            letter=chr(ord("a") + idx),
        )

    for ax in axes[len(curves):]:
        ax.axis("off")

    fig.legend(
        handles=[
            Line2D([0], [0], color=gray_color, lw=2.1, label="ML prediction"),
            Line2D([0], [0], color=blue_color, lw=2.4, label="Fitted curve"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 1.02),
        prop={"size": 14, "weight": "bold"},
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(
        os.path.join(output_dir, PDP_PLOT_FILENAME),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    _save_single_pdp_figures(curves, output_dir, gray_color, blue_color)

    legacy_plot = os.path.join(output_dir, LEGACY_PDP_PLOT_FILENAME)
    if os.path.exists(legacy_plot):
        os.remove(legacy_plot)


def _plot_shap(model, X_sample, feature_names, output_dir):
    if shap is None:
        print("[SHAP] shap not installed; skipping SHAP plots.")
        return

    warnings.filterwarnings("ignore", category=UserWarning)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_bar.png"), dpi=150)
    plt.close()


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"Data file not found: {DATA_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_excel(DATA_PATH)
    (
        df,
        target_col,
        feature_cols,
        binary_cols,
        group1_cols,
        group2_cols,
        temp_col,
    ) = _prepare_features(df)

    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    X = df[feature_cols]
    y = df[target_col]

    numeric_cols = [c for c in feature_cols if c not in binary_cols]

    seed_records = []
    representative_result = None
    for seed in RANDOM_SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        pipeline = _build_pipeline(numeric_cols, binary_cols, random_state=seed)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_pred_train = pipeline.predict(X_train)
        r2_train = r2_score(y_train, y_pred_train)
        r2, rmse, mae, mape = _evaluate(y_test, y_pred)
        seed_records.append(
            {
                "seed": seed,
                "train_r2": r2_train,
                "test_r2": r2,
                "test_rmse": rmse,
                "test_mae": mae,
                "test_mape": mape,
            }
        )
        if seed == REPRESENTATIVE_SEED:
            representative_result = (
                pipeline,
                X_train,
                X_test,
                y_train,
                y_test,
                y_pred_train,
                y_pred,
                r2_train,
                r2,
            )

    seed_results = pd.DataFrame(seed_records)
    seed_results.to_csv(
        os.path.join(OUTPUT_DIR, "metrics_catboost_multi_seed.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print("== Test Metrics Across Random Seeds ==")
    print(seed_results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n== Multi-seed Summary (mean +/- std) ==")
    print(seed_results.drop(columns="seed").agg(["mean", "std"]).round(6))

    (
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred_train,
        y_pred,
        r2_train,
        r2,
    ) = representative_result

    kf = KFold(n_splits=5, shuffle=True, random_state=REPRESENTATIVE_SEED)
    cv_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
    cv_rmse = -cross_val_score(
        pipeline, X, y, cv=kf, scoring="neg_root_mean_squared_error"
    )

    _plot_residuals(y_test, y_pred, OUTPUT_DIR)
    _plot_parity_with_marginals(
        y_train.to_numpy(),
        y_pred_train,
        y_test.to_numpy(),
        y_pred,
        r2_train,
        r2,
        OUTPUT_DIR,
    )

    print("== Stability (5-fold CV) ==")
    print(f"R2 mean: {cv_r2.mean():.6f}, std: {cv_r2.std():.6f}")
    print(f"RMSE mean: {cv_rmse.mean():.6f}, std: {cv_rmse.std():.6f}")

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = _feature_names_from_preprocess(preprocess, numeric_cols, binary_cols)

    _plot_feature_importance(model, feature_names, OUTPUT_DIR)

    pdp_features = _resolve_pdp_features(X_train.columns)
    if pdp_features:
        _plot_pdp(pipeline, X_train, pdp_features, OUTPUT_DIR)

    X_train_proc = preprocess.transform(X_train)
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
    _plot_shap(model, X_train_proc, feature_names, OUTPUT_DIR)


if __name__ == "__main__":
    main()






