import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from json import loads
import streamlit.components.v1 as components
import streamlit_plotly_events as plotly_events_pkg

PLOTLY_EVENTS_BUILD_DIR = Path(plotly_events_pkg.__file__).resolve().parent / "frontend" / "build"
_plotly_events_hover_component = components.declare_component(
    "plotly_events_hover_v3",
    path=str(PLOTLY_EVENTS_BUILD_DIR),
)


def plotly_events(
    plot_fig,
    click_event=True,
    select_event=False,
    hover_event=False,
    override_height=450,
    override_width="100%",
    key=None,
):
    component_value = _plotly_events_hover_component(
        plot_obj=plot_fig.to_json(),
        override_height=override_height,
        override_width=override_width,
        key=key,
        click_event=click_event,
        select_event=select_event,
        hover_event=hover_event,
        default="[]",
    )
    return loads(component_value)

# ── config ────────────────────────────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT  = DASHBOARD_DIR.parents[1]

st.set_page_config(
    page_title="Fleet Health Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# ── data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_predictions():
    return pd.read_csv(DASHBOARD_DIR / "predictions.csv")

@st.cache_data
def load_sensor_data():
    return pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "test_FD001.txt",
        sep=r'\s+',
        header=None,
        names=["unit number", "time, in cycles",
               "operational setting 1", "operational setting 2", "operational setting 3",
               "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
               "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"]
    )

def classify(rul):
    if rul < 30:   return "Critical"
    elif rul < 70: return "Warning"
    else:          return "Healthy"


df_pred   = load_predictions()
df_sensor = load_sensor_data()

df_pred["status"] = df_pred["predicted_rul"].apply(classify)

status_filter = st.sidebar.multiselect(
    "Filter by status",
    options=["Critical", "Warning", "Healthy"],
    default=["Critical", "Warning", "Healthy"]
)

df_filtered = df_pred[df_pred["status"].isin(status_filter)]

# ── navigation ────────────────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Fleet Overview"

page_options = ["Fleet Overview", "Engine Inspector", "Model Performance"]
page = st.sidebar.selectbox(
    "View",
    page_options,
    index=page_options.index(st.session_state["current_page"]),
)
st.session_state["current_page"] = page

if page == "Fleet Overview":
    st.title("⚙️ Fleet Health Dashboard")
    st.caption("NASA CMAPSS FD001 — 100 test engines")

    n_critical = (df_pred["status"] == "Critical").sum()
    n_warning  = (df_pred["status"] == "Warning").sum()
    n_healthy  = (df_pred["status"] == "Healthy").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Critical", n_critical, help="Predicted RUL < 30 cycles")
    col2.metric("🟡 Warning",  n_warning,  help="Predicted RUL 30–70 cycles")
    col3.metric("🟢 Healthy",  n_healthy,  help="Predicted RUL > 70 cycles")

    st.divider()
    sort_by = st.sidebar.radio(
        "Sort fleet bars by",
        options=["Engine ID", "Predicted RUL"],
        horizontal=True,
    )
    sort_direction = st.sidebar.radio(
        "Sort direction",
        options=["Ascending", "Descending"],
        horizontal=True,
    )
    sort_column = "unit_number" if sort_by == "Engine ID" else "predicted_rul"
    df_sorted = df_filtered.sort_values(
        sort_column,
        ascending=(sort_direction == "Ascending"),
    )

    row_height = 18
    chart_margin_height = 120
    color_map = {"Critical": "#e74c3c", "Warning": "#f39c12", "Healthy": "#2ecc71"}

    df_plot = df_sorted.reset_index(drop=True).copy()
    df_plot["engine_label"] = df_plot["unit_number"].astype(str)
    engine_order = df_plot["engine_label"].tolist()

    if df_plot.empty:
        st.info("No engines match the selected status filter.")
        st.stop()

    st.markdown(
        """
        <style>
        div[data-testid="stPlotlyChart"] .barlayer .point {
            transition: filter 120ms ease, stroke-width 120ms ease, stroke 120ms ease;
            cursor: pointer;
        }

        div[data-testid="stPlotlyChart"] .barlayer .point:hover {
            filter: brightness(1.16) drop-shadow(0 2px 5px rgba(0, 0, 0, 0.35));
            stroke: rgba(20, 20, 20, 0.85) !important;
            stroke-width: 2px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_plot["predicted_rul"].astype(float).tolist(),
            y=df_plot["engine_label"].tolist(),
            orientation="h",
            marker_color=df_plot["status"].map(color_map).tolist(),
            marker_line=dict(color="rgba(30, 30, 30, 0.35)", width=0.5),
            customdata=df_plot[["unit_number", "status"]].values.tolist(),
            hovertemplate=(
                "Engine ID=%{customdata[0]}<br>"
                "Predicted RUL=%{x:.1f} cycles<br>"
                "Status=%{customdata[1]}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    for status, color in color_map.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=status,
            )
        )
    fig.update_layout(
        height=max(chart_margin_height + row_height, chart_margin_height + len(df_plot) * row_height),
        xaxis_title="Predicted RUL (cycles)",
        yaxis_title="Engine ID",
        margin=dict(t=30, b=60),
        font=dict(color="rgba(250, 250, 250, 0.88)", size=13),
        legend=dict(
            font=dict(color="rgba(250, 250, 250, 0.88)", size=13),
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="rgba(30, 30, 30, 0.6)",
            font_size=13,
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        yaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=engine_order,
            autorange="reversed",
            automargin=True,
            tickfont=dict(color="rgba(250, 250, 250, 0.82)", size=12),
            title_font=dict(color="rgba(250, 250, 250, 0.9)", size=14),
        ),
        xaxis=dict(
            range=[0, df_plot["predicted_rul"].max() * 1.05],
            automargin=True,
            tickfont=dict(color="rgba(250, 250, 250, 0.82)", size=12),
            title_font=dict(color="rgba(250, 250, 250, 0.9)", size=14),
            gridcolor="rgba(160, 160, 160, 0.18)",
            gridwidth=0.5,
            zeroline=False,
        ),
        bargap=0.1
    )


    selected_points = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=fig.layout.height,
        key="fleet_overview_barplot_hover_v3",
    )
    if selected_points:
        point = selected_points[0]
        clicked_engine = point["y"]
        st.session_state["selected_engine"] = int(clicked_engine)
        st.session_state["current_page"] = "Engine Inspector"
        st.rerun()
    st.caption("Select an engine in the Engine Inspector tab to view sensor details.")

elif page == "Engine Inspector":
    st.title("Engine Inspector")

    default_engine = st.session_state.get("selected_engine", 1)
    engine_id = st.sidebar.selectbox(
        "Select Engine",
        options=sorted(df_pred["unit_number"].tolist()),
        index=sorted(df_pred["unit_number"].tolist()).index(default_engine)
    )
    st.session_state["selected_engine"] = engine_id
    # prediction summary
    engine_pred = df_pred[df_pred["unit_number"] == engine_id].iloc[0]
    status = engine_pred["status"]
    color  = {"Critical": "#e74c3c", "Warning": "#f39c12", "Healthy": "#2ecc71"}[status]

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted RUL", f"{engine_pred['predicted_rul']:.1f} cycles")
    col2.metric("Actual RUL",    f"{engine_pred['actual_rul']:.1f} cycles")
    col3.metric("Error",         f"{engine_pred['predicted_rul'] - engine_pred['actual_rul']:.1f} cycles")

    st.markdown(f"<span style='color:{color}; font-size:1.2em; font-weight:bold'>● {status}</span>", 
            unsafe_allow_html=True)
    st.divider()

    # sensor plots
    engine_sensors = df_sensor[df_sensor["unit number"] == engine_id]

    # the 5 most informative sensors based on EDA    
    st.subheader("Sensor Trends")

    sensors_to_plot = ["s7", "s11", "s12", "s15", "s20"]

    # normalise each sensor to 0-1 range for combined view
    engine_norm = engine_sensors.copy()
    for s in sensors_to_plot:
        min_v = engine_norm[s].min()
        max_v = engine_norm[s].max()
        engine_norm[s] = (engine_norm[s] - min_v) / (max_v - min_v + 1e-8)


    sensors_to_plot = ["s7", "s11", "s12", "s15", "s20"]
    sensor_colors = {
        "s7":  "#636EFA",
        "s11": "#EF553B", 
        "s12": "#00CC96",
        "s15": "#AB63FA",
        "s20": "#FFA15A"
    }

    fig_norm = px.line(
        engine_norm,
        x="time, in cycles",
        y=sensors_to_plot,
        color_discrete_map=sensor_colors,
        labels={"time, in cycles": "Cycle", "value": "Normalised Value", "variable": "Sensor"},
        title="Normalised Sensor Overview",
    )
    st.plotly_chart(fig_norm, width='stretch')

    # individual plots
    st.subheader("Individual Sensor Detail")
    cols = st.columns(2)
    for i, sensor in enumerate(sensors_to_plot):
        fig_s = px.line(
            engine_sensors,
            x="time, in cycles",
            y=sensor,
            title=sensor,
            labels={"time, in cycles": "Cycle"},
        )
        fig_s.update_traces(line_color=sensor_colors[sensor])
        cols[i % 2].plotly_chart(fig_s, width='stretch')

elif page == "Model Performance":
    st.title("Model Performance")
    st.caption("Evaluated on 100 held-out test engines — NASA CMAPSS FD001")

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np

    y_true = df_pred["actual_rul"].values
    y_pred_vals = df_pred["predicted_rul"].values
    rmse   = mean_squared_error(y_true, y_pred_vals) ** 0.5
    mae    = mean_absolute_error(y_true, y_pred_vals)
    d      = y_pred_vals - y_true
    cmapss = float(np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)))

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.1f} cycles")
    col2.metric("MAE",  f"{mae:.1f} cycles")
    col3.metric("CMAPSS Score", f"{cmapss:.0f}")

    st.divider()

    # ── 1. Model comparison table ─────────────────────────────────────────────
    st.subheader("Model Comparison")
    comparison = pd.DataFrame({
        "Model":         ["Random Forest", "XGBoost", "LSTM (tuned)"],
        "RMSE":          [19.6, 19.7, 21.2],
        "MAE":           [14.4, 14.8, 16.0],
        "CMAPSS Score":  [1142, 1199, 1093],
    })
    st.dataframe(comparison, hide_index=True, width='stretch')
    st.caption("LSTM wins on CMAPSS — fewer dangerous overestimates despite slightly higher RMSE.")

    st.divider()

    col_left, col_right = st.columns(2)

    # ── scatter: predicted vs actual ──────────────────────────────────────────
    with col_left:
        fig_scatter = px.scatter(
            df_pred,
            x="actual_rul",
            y="predicted_rul",
            color="status",
            color_discrete_map={"Critical": "#e74c3c", "Warning": "#f39c12", "Healthy": "#2ecc71"},
            labels={"actual_rul": "Actual RUL", "predicted_rul": "Predicted RUL"},
            title="Predicted vs Actual RUL",
            hover_data=["unit_number"],
        )
        max_rul = max(y_true.max(), y_pred_vals.max())
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_rul], y=[0, max_rul],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Perfect prediction"
        ))
        st.plotly_chart(fig_scatter, width='stretch')

    # ── residual histogram ────────────────────────────────────────────────────
    with col_right:
        residuals = y_pred_vals - y_true
        fig_resid = px.histogram(
            x=residuals,
            nbins=30,
            labels={"x": "Residual (predicted − actual)"},
            title="Residual Distribution",
            color_discrete_sequence=["#3498db"]
        )
        fig_resid.add_vline(x=0, line_dash="dash", line_color="red")
        fig_resid.add_annotation(
            x=0, y=1, yref="paper",
            text="← underestimate | overestimate →",
            showarrow=False, yshift=10
        )
        st.plotly_chart(fig_resid, width='stretch')

    st.divider()

    # ── 2. Error by RUL range ─────────────────────────────────────────────────
    st.subheader("Mean Absolute Error by RUL Range")
    # top edge follows the data: actual RUL reaches 145 here, so a hard 125 cap
    # would silently drop the 11 longest-lived engines from this chart
    bins   = [0, 30, 70, max(125.0, float(df_pred["actual_rul"].max()))]
    labels = ["Critical (0–30)", "Warning (30–70)", "Healthy (70+)"]
    df_pred["rul_range"] = pd.cut(
        df_pred["actual_rul"], bins=bins, labels=labels, include_lowest=True
    )
    df_pred["abs_error"] = (df_pred["predicted_rul"] - df_pred["actual_rul"]).abs()

    mae_by_range = df_pred.groupby("rul_range", observed=True)["abs_error"].mean().reset_index()
    fig_range = px.bar(
        mae_by_range,
        x="rul_range",
        y="abs_error",
        color="rul_range",
        color_discrete_map={
            "Critical (0–30)":  "#e74c3c",
            "Warning (30–70)":  "#f39c12",
            "Healthy (70+)":    "#2ecc71",
        },
        labels={"rul_range": "RUL Range", "abs_error": "Mean Absolute Error (cycles)"},
        title="Where does the model struggle most?",
    )
    fig_range.update_layout(showlegend=False)
    st.plotly_chart(fig_range, width='stretch')

    st.divider()

    # ── 3. Worst predictions ──────────────────────────────────────────────────
    st.subheader("Worst Predictions")
    df_pred["error"] = df_pred["predicted_rul"] - df_pred["actual_rul"]
    worst = df_pred.reindex(df_pred["error"].abs().sort_values(ascending=False).index).head(10)
    worst_display = worst[["unit_number", "actual_rul", "predicted_rul", "error", "status"]].copy()
    worst_display.columns = ["Engine", "Actual RUL", "Predicted RUL", "Error", "Status"]
    worst_display = worst_display.reset_index(drop=True)

    def color_error(val):
        if val > 20:  return "color: #e74c3c"
        if val < -20: return "color: #f39c12"
        return ""

    st.dataframe(
        worst_display.style.map(color_error, subset=["Error"]),
        hide_index=True,
        width='stretch'
    )
    st.caption("Positive error = overestimate (predicted more life remaining than actual) — the more dangerous failure mode.")
