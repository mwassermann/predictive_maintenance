"""Export publication-quality figures for the project showcase.

Writes every figure twice into reports/figures/:
  *.png  - 200 dpi, for the GitHub README
  *.pdf  - vector, for slides

Run with:  python -m src.viz.export_figures
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR  = PROJECT_ROOT / "data" / "raw"
PRED_CSV = PROJECT_ROOT / "src" / "dashboard" / "predictions.csv"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
OUT_DIR  = PROJECT_ROOT / "reports" / "figures"

# ── design tokens ─────────────────────────────────────────────────────────────
# Status colours are reserved for engine condition and never reused as series
# colours; series identity uses the categorical slots in fixed order.
SURFACE  = "#fcfcfb"
INK      = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
BASELINE = "#c3c2b7"

STATUS = {"Critical": "#d03b3b", "Warning": "#fab219", "Healthy": "#0ca30c"}
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# diverging poles for signed residuals (blue <-> red, neutral midpoint)
DIV_UNDER = "#2a78d6"   # predicted less life than reality - conservative
DIV_OVER  = "#e34948"   # predicted more life than reality - dangerous

SENSORS = ["s7", "s11", "s12", "s15", "s20"]   # most informative per EDA
RUL_CAP = 125                                   # training labels are capped here

COLUMNS = (["unit", "cycle", "os1", "os2", "os3"]
           + [f"s{i}" for i in range(1, 22)])


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": BASELINE,
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.labelsize": 11,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "pdf.fonttype": 42,     # embed TrueType so slide text stays selectable
        "svg.fonttype": "none",
    })


def _style_axes(ax, *, grid_axis: str = "y") -> None:
    """Recessive chrome: no box, hairline grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, axis=grid_axis, linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in (("png", {"dpi": 200}), ("pdf", {})):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"  wrote {name}.png + {name}.pdf")


def _footnote(fig, text: str, dy: float = -0.04) -> None:
    fig.text(0.0, dy, text, ha="left", va="top", fontsize=9, color=MUTED)


# ── data loaders ──────────────────────────────────────────────────────────────
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PRED_CSV)
    df["error"] = df["predicted_rul"] - df["actual_rul"]
    df["abs_error"] = df["error"].abs()
    df["status"] = pd.cut(
        df["predicted_rul"],
        bins=[-np.inf, 30, 70, np.inf],
        labels=["Critical", "Warning", "Healthy"],
    )
    return df


def load_model_comparison() -> pd.DataFrame:
    """Test-set metrics, read from the MLflow runs when the DB is present.

    mlflow.db is gitignored, so fall back to the values those runs produced -
    the same figures quoted in the README.
    """
    fallback = pd.DataFrame({
        "model":  ["Random Forest", "XGBoost", "LSTM (tuned)"],
        "rmse":   [19.6, 19.7, 21.2],
        "cmapss": [1141.5, 1199.1, 1093.4],
    })
    if not MLFLOW_DB.exists():
        print("  (mlflow.db absent - using recorded fallback metrics)")
        return fallback

    # mlflow.db holds more than one run per model name (an earlier untuned
    # LSTM evaluation sits alongside the final one) - always take the most
    # recent run per model, never an arbitrary row.
    query = """
        SELECT r.name,
               MAX(r.start_time) AS start_time,
               MAX(CASE WHEN m.key = 'rmse'   THEN m.value END) AS rmse,
               MAX(CASE WHEN m.key = 'cmapss' THEN m.value END) AS cmapss
        FROM runs r JOIN metrics m ON m.run_uuid = r.run_uuid
        WHERE r.name LIKE '%-test-evaluation' AND r.status = 'FINISHED'
        GROUP BY r.run_uuid
        HAVING cmapss IS NOT NULL AND rmse IS NOT NULL
    """
    with sqlite3.connect(MLFLOW_DB) as conn:
        runs = pd.read_sql_query(query, conn)

    labels = {"rf-test-evaluation": "Random Forest",
              "xgboost-test-evaluation": "XGBoost",
              "lstm-test-evaluation": "LSTM (tuned)"}
    runs["model"] = runs["name"].map(labels)
    runs = (runs.dropna(subset=["model"])
                .sort_values("start_time", ascending=False)
                .drop_duplicates("model", keep="first"))

    if set(runs["model"]) != set(fallback["model"]):
        print("  (incomplete MLflow runs - using recorded fallback metrics)")
        return fallback

    order = {m: i for i, m in enumerate(fallback["model"])}
    runs = (runs.sort_values("model", key=lambda s: s.map(order))
                .reset_index(drop=True)[["model", "rmse", "cmapss"]])

    # Guard against silently picking up a stale run: the selected LSTM must
    # agree with the predictions.csv the dashboard and figures 02/04/05 use.
    expected = fallback.set_index("model")
    drift = [
        f"{row.model} {col} {row[col]:.1f} vs expected {expected.loc[row.model, col]:.1f}"
        for row in runs.itertuples()
        for col in ("rmse", "cmapss")
        if abs(getattr(row, col) - expected.loc[row.model, col]) > 0.5
    ]
    if drift:
        print("  WARNING: MLflow metrics differ from the recorded run:")
        for line in drift:
            print(f"    - {line}")
        print("    Retrain or update the fallback if this is intentional.")

    print("  (metrics read from mlflow.db)")
    return runs


def load_training_sensors() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "train_FD001.txt", sep=r"\s+",
                     header=None, names=COLUMNS)
    df["rul"] = df["unit"].map(df.groupby("unit")["cycle"].max()) - df["cycle"]
    return df


# ── figures ───────────────────────────────────────────────────────────────────
def fig_degradation_signal(train: pd.DataFrame) -> None:
    """Why RUL is predictable at all: every informative sensor drifts.

    Small multiples rather than one overlaid panel - the five sensors live on
    different scales, and normalising them onto a shared axis pins them all to
    0 and 1 at the extremes, where no direct label can be placed legibly.
    """
    sub = train[train["rul"] <= 200]
    counts = sub.groupby("rul").size()
    keep = counts[counts >= 80].index          # drop the thin long-life tail
    profile = sub[sub["rul"].isin(keep)].groupby("rul")[SENSORS].mean()
    profile = profile.rolling(7, center=True, min_periods=1).mean()

    fig, axes = plt.subplots(1, len(SENSORS), figsize=(13.0, 3.4), sharex=True)
    for ax, sensor, colour in zip(axes, SENSORS, CATEGORICAL):
        ax.plot(profile.index, profile[sensor], color=colour, linewidth=2.0)
        ax.axvline(RUL_CAP, color=BASELINE, linestyle=(0, (4, 3)), linewidth=1.1)
        ax.set_title(sensor, fontsize=12, color=colour, fontweight="semibold")
        ax.invert_xaxis()                       # time flows left to right
        ax.set_xticks([150, 100, 50, 0])
        ax.tick_params(axis="y", labelsize=9)
        _style_axes(ax, grid_axis="both")

    axes[0].set_ylabel("Sensor reading")
    axes[len(SENSORS) // 2].set_xlabel("Remaining useful life (cycles)")
    fig.suptitle("Every informative sensor drifts monotonically toward failure",
                 fontsize=14, fontweight="semibold", color=INK, y=1.06)
    _footnote(fig, f"Mean across {train['unit'].nunique()} training engines (NASA "
                   "CMAPSS FD001), smoothed over 7 cycles; each panel keeps its own "
                   "raw scale. Failure is at RUL = 0, on the right of every panel.")
    _footnote(fig, f"Dashed line: the {RUL_CAP}-cycle cap applied to training "
                   "labels. The drift is gentle above it and steepens below.", dy=-0.13)
    _save(fig, "01_degradation_signal")


def fig_predicted_vs_actual(pred: pd.DataFrame) -> None:
    """Does it work? One scatter against the identity line."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    limit = max(pred["actual_rul"].max(), pred["predicted_rul"].max()) * 1.06

    ax.plot([0, limit], [0, limit], color=BASELINE,
            linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax.annotate("perfect prediction",
                xy=(limit * 0.90, limit * 0.90), xycoords="data",
                xytext=(limit * 0.58, limit * 0.97), textcoords="data",
                fontsize=9.5, color=MUTED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=BASELINE, linewidth=0.9))

    for status, colour in STATUS.items():
        grp = pred[pred["status"] == status]
        ax.scatter(grp["actual_rul"], grp["predicted_rul"], s=54,
                   color=colour, edgecolor=SURFACE, linewidth=1.2,
                   label=f"{status}  (n={len(grp)})", zorder=3)

    ceiling = pred["predicted_rul"].max()
    ax.axhline(ceiling, color=MUTED, linewidth=0.9, linestyle=(0, (2, 3)), zorder=2)
    ax.annotate(f"model never predicts above {ceiling:.0f} cycles\n"
                "— a direct consequence of the 125-cycle label cap",
                xy=(limit * 0.02, ceiling), xytext=(0, 9),
                textcoords="offset points", fontsize=9.5, color=INK_SOFT)

    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title("Predicted vs actual RUL, 100 held-out engines")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.legend(title="Predicted status", title_fontsize=10,
              loc="lower right", labelcolor=INK_SOFT)
    _style_axes(ax, grid_axis="both")
    _footnote(fig, "Points below the dashed line are underestimates (early "
                   "maintenance); points above are overestimates (late).")
    _save(fig, "02_predicted_vs_actual")


def fig_rmse_vs_cmapss(models: pd.DataFrame) -> None:
    """The project's headline finding: the ranking inverts between metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    colours = CATEGORICAL[:len(models)]
    positions = np.arange(len(models))

    panels = [
        (axes[0], "rmse",   "RMSE (cycles)",  "RMSE — symmetric error",       "{:.1f}"),
        (axes[1], "cmapss", "CMAPSS score",   "CMAPSS — asymmetric cost",     "{:,.0f}"),
    ]
    for ax, column, ylabel, title, fmt in panels:
        values = models[column].to_numpy()
        ax.bar(positions, values, color=colours, width=0.6, zorder=3)
        best = int(np.argmin(values))
        for i, value in enumerate(values):
            ax.annotate(fmt.format(value), xy=(i, value), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        fontsize=10.5, color=INK,
                        fontweight="semibold" if i == best else "normal")
        ax.annotate("best", xy=(best, 0), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=9.5, color=SURFACE, fontweight="semibold")
        ax.set_xticks(positions)
        ax.set_xticklabels(models["model"], fontsize=10, color=INK_SOFT)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12.5)
        ax.set_ylim(0, values.max() * 1.18)
        _style_axes(ax)

    fig.suptitle("The same three models rank differently depending on the metric",
                 fontsize=14, fontweight="semibold", color=INK, y=1.02)
    _footnote(fig, "CMAPSS penalises overestimated RUL exponentially harder than "
                   "underestimated RUL. The LSTM has the worst RMSE of the three "
                   "and the best CMAPSS score —\nit makes larger errors, but fewer "
                   "of the ones that let an engine run past failure.")
    _save(fig, "03_rmse_vs_cmapss")


def fig_residual_distribution(pred: pd.DataFrame) -> None:
    """Where the errors fall: conservative on the left, dangerous on the right."""
    residuals = pred["error"].to_numpy()
    span = np.abs(residuals).max() * 1.08
    edges = np.linspace(-span, span, 29)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    counts, _ = np.histogram(residuals, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    colours = [DIV_UNDER if c < 0 else DIV_OVER for c in centres]
    ax.bar(centres, counts, width=np.diff(edges) * 0.9,
           color=colours, zorder=3)

    ax.axvline(0, color=INK_SOFT, linewidth=1.1, zorder=4)
    n_under = int((residuals < 0).sum())
    n_over  = int((residuals > 0).sum())
    top = counts.max() * 1.22
    ax.set_ylim(0, top)

    for x, text, colour in (
        (-span * 0.55, f"underestimate — {n_under} engines\nserviced early, costs money",
         DIV_UNDER),
        (span * 0.55,  f"overestimate — {n_over} engines\nserviced late, costs an engine",
         DIV_OVER),
    ):
        ax.annotate(text, xy=(x, top * 0.88), ha="center", va="top",
                    fontsize=10, color=colour, fontweight="semibold",
                    linespacing=1.5)

    ax.set_xlabel("Residual: predicted − actual RUL (cycles)")
    ax.set_ylabel("Engines")
    ax.set_title("Residual distribution — the asymmetry CMAPSS is built to punish")
    _style_axes(ax)
    _footnote(fig, f"Mean residual {residuals.mean():+.1f} cycles, "
                   f"median {np.median(residuals):+.1f}. "
                   "A symmetric metric like RMSE scores both sides identically.")
    _save(fig, "04_residual_distribution")


def fig_mae_by_rul_range(pred: pd.DataFrame) -> None:
    """Accuracy where it matters: the model is sharpest near failure."""
    top_edge = max(float(RUL_CAP), float(pred["actual_rul"].max()))
    labels = ["Critical\n(0–30)", "Warning\n(30–70)", "Healthy\n(70+)"]
    pred = pred.copy()
    pred["band"] = pd.cut(pred["actual_rul"], bins=[0, 30, 70, top_edge],
                          labels=labels, include_lowest=True)
    grouped = pred.groupby("band", observed=True)["abs_error"].agg(["mean", "size"])
    grouped = grouped.reindex(labels)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    positions = np.arange(len(labels))
    ax.bar(positions, grouped["mean"], width=0.6, zorder=3,
           color=[STATUS["Critical"], STATUS["Warning"], STATUS["Healthy"]])
    for i, (mean, size) in enumerate(zip(grouped["mean"], grouped["size"])):
        ax.annotate(f"{mean:.1f}", xy=(i, mean), xytext=(0, 7),
                    textcoords="offset points", ha="center",
                    fontsize=12, fontweight="semibold", color=INK)
        ax.annotate(f"n={int(size)}", xy=(i, mean), xytext=(0, -16),
                    textcoords="offset points", ha="center",
                    fontsize=9.5, color=SURFACE)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10.5, color=INK_SOFT)
    ax.set_xlabel("Actual RUL band (cycles)")
    ax.set_ylabel("Mean absolute error (cycles)")
    ax.set_title("The model is most accurate on the engines closest to failure")
    ax.set_ylim(0, grouped["mean"].max() * 1.22)
    _style_axes(ax)
    _footnote(fig, "All 100 test engines are covered; the top band is open-ended "
                   f"because actual RUL reaches {pred['actual_rul'].max():.0f} cycles.")
    _save(fig, "05_mae_by_rul_range")


def main() -> None:
    _apply_style()
    print(f"Exporting figures to {OUT_DIR.relative_to(PROJECT_ROOT)}/")

    predictions = load_predictions()
    models = load_model_comparison()

    fig_predicted_vs_actual(predictions)
    fig_rmse_vs_cmapss(models)
    fig_residual_distribution(predictions)
    fig_mae_by_rul_range(predictions)

    train_path = RAW_DIR / "train_FD001.txt"
    if train_path.exists():
        fig_degradation_signal(load_training_sensors())
    else:
        print(f"  skipped 01_degradation_signal ({train_path} not found)")

    print("Done.")


if __name__ == "__main__":
    main()
