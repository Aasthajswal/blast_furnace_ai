import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product as iproduct

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor,
                               GradientBoostingRegressor,
                               ExtraTreesRegressor)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════
st.set_page_config(
    page_title="Blast Furnace AI",
    page_icon="🔥",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Share+Tech+Mono&display=swap');

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes countUp {
    from { opacity: 0; transform: scale(0.85); }
    to   { opacity: 1; transform: scale(1); }
}
section.main > div { animation: fadeSlideIn 0.7s ease both; }
div[data-testid="stVerticalBlock"] > div:nth-child(1) { animation-delay: 0.05s; }
div[data-testid="stVerticalBlock"] > div:nth-child(2) { animation-delay: 0.12s; }
div[data-testid="stVerticalBlock"] > div:nth-child(3) { animation-delay: 0.19s; }
div[data-testid="stVerticalBlock"] > div:nth-child(4) { animation-delay: 0.26s; }
h1 { font-family: 'Rajdhani', sans-serif !important; letter-spacing: 1px; }
h3, h4 { font-family: 'Rajdhani', sans-serif !important; }
div[data-testid="stNumberInput"] input {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# DATA GENERATION  (exact same as generate_data.py)
# ══════════════════════════════════════════
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'blast_temp':        np.random.uniform(900,  1200, n),
        'coke_rate':         np.random.uniform(300,   600, n),
        'pci_rate':          np.random.uniform(100,   250, n),
        'oxygen_enrichment': np.random.uniform(2,       6, n),
        'air_flow':          np.random.uniform(3000, 5000, n),
        'burden_iron_ore':   np.random.uniform(60,     80, n),
        'moisture':          np.random.uniform(1,       5, n),
    })
    df['hot_metal_output'] = (
        2500
        + 0.4   * df['blast_temp']
        - 0.3   * df['coke_rate']
        - 0.1   * df['pci_rate']
        + 0.5   * df['oxygen_enrichment']
        + 0.002 * df['air_flow']
        + 0.3   * df['burden_iron_ore']
        - 1.5   * df['moisture']
        + np.random.normal(0, 20, n)
    )
    df['fuel_consumption'] = (
        200
        + 0.6   * df['coke_rate']
        + 0.4   * df['pci_rate']
        - 0.02  * df['blast_temp']
        - 0.5   * df['oxygen_enrichment']
        - 0.01  * df['air_flow']
        + np.random.normal(0, 15, n)
    )
    df['thermal_efficiency'] = np.clip(
        0.05 * df['blast_temp']
        - 0.03 * df['coke_rate']
        + 0.005 * df['air_flow']
        + 0.8  * df['oxygen_enrichment']
        - 0.4  * df['moisture']
        + np.random.normal(0, 3, n),
        40, 90
    )
    return df


# ══════════════════════════════════════════
# MODEL TRAINING  (exact same as model.py)
# ══════════════════════════════════════════
@st.cache_resource
def train_all_models():
    df = generate_data()
    FEATURE_COLS = ['blast_temp', 'coke_rate', 'pci_rate',
                    'oxygen_enrichment', 'air_flow', 'burden_iron_ore', 'moisture']
    TARGET_COLS  = ['hot_metal_output', 'fuel_consumption', 'thermal_efficiency']

    X = df[FEATURE_COLS]
    y = df[TARGET_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models_def = {
        "Linear Regression":      LinearRegression(),
        "Random Forest":           RandomForestRegressor(n_estimators=100, random_state=42),
        "Extra Trees":             ExtraTreesRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":       MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                      max_depth=3, random_state=42)),
        "Polynomial Regression":   Pipeline([
            ('poly',   PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())]),
        "XGBoost":                 MultiOutputRegressor(
            XGBRegressor(n_estimators=100, learning_rate=0.1,
                         max_depth=4, random_state=42, verbosity=0)),
    }

    results_rows = []
    trained = {}
    for name, mdl in models_def.items():
        mdl.fit(X_train, y_train)
        trained[name] = mdl
        y_pred = mdl.predict(X_test)
        r2s   = [r2_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(3)]
        rmses = [np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
                 for i in range(3)]
        results_rows.append({
            "Model": name,
            "Average R²":   round(float(np.mean(r2s)),   4),
            "Average RMSE": round(float(np.mean(rmses)),  4),
        })

    results_df = pd.DataFrame(results_rows).sort_values(
        "Average R²", ascending=False).reset_index(drop=True)

    best_model = trained["Linear Regression"]   # confirmed best from model.py
    rf_model   = trained["Random Forest"]

    # Feature importances — RF is MultiOutput so average across estimators
    if hasattr(rf_model, 'estimators_'):
        feat_imp = rf_model.feature_importances_
    else:
        feat_imp = np.mean([e.feature_importances_
                            for e in rf_model.estimators_], axis=0)

    # Actual vs Predicted arrays for LR (for scatter plots)
    y_pred_lr = best_model.predict(X_test)

    return best_model, rf_model, results_df, feat_imp, FEATURE_COLS, \
           X_test, y_test, y_pred_lr


best_model, rf_model, results, feat_imp, feature_cols, \
    X_test, y_test, y_pred_lr = train_all_models()

# ── Real averages from generate_data.py output ──
AVG_HOT_METAL  = 2791.40
AVG_FUEL       = 478.38
AVG_EFFICIENCY = 60.76


# ══════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════
def number_slider(label, min_val, max_val, default, step=1.0, key=None):
    sk = f"sl_{key}"
    nk = f"ni_{key}"
    if sk not in st.session_state:
        st.session_state[sk] = float(default)
    if nk not in st.session_state:
        st.session_state[nk] = float(default)

    def on_slider():
        st.session_state[nk] = st.session_state[sk]

    def on_number():
        val = max(float(min_val), min(float(max_val), float(st.session_state[nk])))
        st.session_state[sk] = val
        st.session_state[nk] = val

    col_s, col_n = st.columns([3, 1])
    col_s.slider(label, float(min_val), float(max_val),
                 key=sk, step=float(step), on_change=on_slider)
    col_n.number_input("✏️", min_value=float(min_val), max_value=float(max_val),
                       key=nk, step=float(step), on_change=on_number,
                       label_visibility="visible")
    return float(st.session_state[sk])


def draw_gauge(value, min_val, max_val, label, good_range, warn_range, unit="%"):
    fig, ax = plt.subplots(figsize=(4, 2.4), subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    total = max_val - min_val

    def val_to_angle(v):
        return 180 - (v - min_val) / total * 180

    zones = [
        (min_val,       warn_range[0], '#e74c3c'),
        (warn_range[0], good_range[0], '#f39c12'),
        (good_range[0], good_range[1], '#2ecc71'),
        (good_range[1], warn_range[1], '#f39c12'),
        (warn_range[1], max_val,       '#e74c3c'),
    ]
    for (start, end, color) in zones:
        a1    = np.radians(val_to_angle(end))
        a2    = np.radians(val_to_angle(start))
        theta = np.linspace(a1, a2, 60)
        r_outer, r_inner = 1.0, 0.65
        xs = np.concatenate([r_outer*np.cos(theta), r_inner*np.cos(theta[::-1])])
        ys = np.concatenate([r_outer*np.sin(theta), r_inner*np.sin(theta[::-1])])
        ax.fill(xs, ys, color=color, alpha=0.88)

    needle_angle = np.radians(val_to_angle(np.clip(value, min_val, max_val)))
    nx = 0.82 * np.cos(needle_angle)
    ny = 0.82 * np.sin(needle_angle)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color='white',
                                lw=2.2, mutation_scale=14))
    ax.add_patch(plt.Circle((0, 0), 0.07, color='white', zorder=5))
    ax.text(0, -0.28, f"{value:.1f}{unit}", ha='center', va='center',
            fontsize=14, fontweight='bold', color='white', fontfamily='monospace')
    ax.text(0, -0.52, label, ha='center', va='center', fontsize=8, color='#aaaaaa')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.65, 1.15)
    ax.axis('off')
    plt.tight_layout(pad=0.1)
    return fig


def colored_metric(col, title, value_str, delta, delta_good_direction="positive"):
    is_good  = (delta >= 0) if delta_good_direction == "positive" else (delta <= 0)
    arrow    = "▲" if delta >= 0 else "▼"
    color    = "#2ecc71" if is_good else "#e74c3c"
    delta_str = f"{delta:+.1f} vs avg"
    col.markdown(f"""
    <div style="
        background: linear-gradient(135deg,#1a1a2e,#16213e);
        border: 1px solid #2d3561;
        border-radius:12px; padding:16px 20px;
        margin-bottom:4px;
        animation: countUp 0.6s ease both;
    ">
      <div style="font-size:0.78rem;color:#8899aa;
                  font-family:'Rajdhani',sans-serif;letter-spacing:1px;">
        {title}
      </div>
      <div style="font-size:1.55rem;font-weight:700;color:#ffffff;
                  font-family:'Rajdhani',sans-serif;margin:4px 0;">
        {value_str}
      </div>
      <div style="font-size:0.82rem;color:{color};font-weight:600;">
        {arrow} {delta_str}
      </div>
    </div>
    """, unsafe_allow_html=True)


def smart_optimize(current_params, model):
    """
    Grid search matching model.py's optimize_parameters logic.
    Iterates blast_temp × coke_rate × pci_rate (7×7×4 = 196 combos),
    fixing other params near current values.
    Composite score: hot_metal - 0.3*fuel + 0.5*efficiency
    """
    blast_temps = np.arange(900,  1201, 50)   # 7 values
    coke_rates  = np.arange(300,   601, 50)   # 7 values
    pci_rates   = np.arange(100,   251, 50)   # 4 values

    fixed = {
        'oxygen_enrichment': current_params['oxygen_enrichment'],
        'air_flow':          current_params['air_flow'],
        'burden_iron_ore':   current_params['burden_iron_ore'],
        'moisture':          current_params['moisture'],
    }

    best_score  = -np.inf
    best_params = current_params.copy()
    best_pred   = None

    for bt, cr, pr in iproduct(blast_temps, coke_rates, pci_rates):
        row   = pd.DataFrame([{'blast_temp': bt, 'coke_rate': cr,
                                'pci_rate': pr, **fixed}])
        pred  = model.predict(row)[0]
        hm, fuel, eff = pred[0], pred[1], pred[2]
        score = hm - 0.3 * fuel + 0.5 * eff
        if score > best_score:
            best_score  = score
            best_pred   = pred
            best_params = {'blast_temp': bt, 'coke_rate': cr,
                           'pci_rate': pr, **fixed}

    return best_params, best_pred


# ══════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════
st.markdown('<h1 style="font-size:2.1rem;">🔥 Blast Furnace AI Optimization System</h1>',
            unsafe_allow_html=True)
st.markdown("**SAIL Bokaro Steel Plant — Predictive Analytics Dashboard**")
st.markdown("*Trained on synthetic data using domain-realistic ranges from SAIL Bokaro internship*")
st.markdown("---")

# ══════════════════════════════════════════
# TABS
# ══════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ Prediction & Optimization",
    "📊 Model Comparison",
    "📈 Feature Importance",
    "ℹ️ About Project"
])


# ──────────────────────────────────────────
# TAB 1 — Prediction & Optimization
# ──────────────────────────────────────────
with tab1:
    st.subheader("Enter Furnace Parameters")
    st.markdown("Each parameter has a **slider** + **number input** — use whichever is faster.")

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("#### 🎛️ Input Parameters")
        blast_temp        = number_slider("Blast Temperature (°C)",  900,  1200, 1050, step=5.0,  key="bt")
        coke_rate         = number_slider("Coke Rate (kg/ton)",      300,   600,  450, step=5.0,  key="cr")
        pci_rate          = number_slider("PCI Rate (kg/ton)",       100,   250,  175, step=5.0,  key="pr")
        oxygen_enrichment = number_slider("Oxygen Enrichment (%)",   2.0,   6.0,  4.0, step=0.1,  key="oe")
        air_flow          = number_slider("Air Flow (m³/min)",      3000,  5000, 4000, step=50.0, key="af")
        burden_iron_ore   = number_slider("Burden Iron Ore (%)",    60.0,  80.0, 70.0, step=0.5,  key="bi")
        moisture          = number_slider("Moisture (%)",            1.0,   5.0,  3.0, step=0.1,  key="mo")

    current_params = {
        'blast_temp':        blast_temp,
        'coke_rate':         coke_rate,
        'pci_rate':          pci_rate,
        'oxygen_enrichment': oxygen_enrichment,
        'air_flow':          air_flow,
        'burden_iron_ore':   burden_iron_ore,
        'moisture':          moisture,
    }
    prediction = best_model.predict(pd.DataFrame([current_params]))[0]
    hot_metal  = prediction[0]
    fuel       = prediction[1]
    efficiency = prediction[2]

    with col_r:
        st.markdown("#### 📊 Predicted Output")
        m1, m2, m3 = st.columns(3)
        colored_metric(m1, "🏭 Hot Metal Output",  f"{hot_metal:.1f} t/day",
                       hot_metal  - AVG_HOT_METAL,  "positive")
        colored_metric(m2, "⛽ Fuel Consumption",   f"{fuel:.1f} kg/ton",
                       fuel       - AVG_FUEL,        "negative")
        colored_metric(m3, "🌡️ Thermal Efficiency", f"{efficiency:.1f} %",
                       efficiency - AVG_EFFICIENCY,  "positive")

        st.markdown("---")
        st.markdown("#### 🌡️ Thermal Efficiency Gauge")
        gauge_col, _ = st.columns([1.2, 1])
        with gauge_col:
            gfig = draw_gauge(
                value=efficiency, min_val=40, max_val=90,
                label="Thermal Efficiency",
                good_range=(58, 72), warn_range=(50, 80), unit="%"
            )
            st.pyplot(gfig, width="content")
            plt.close()

        st.markdown("---")
        st.markdown("#### 💡 Recommendations")
        recs = []
        if coke_rate > 480:
            recs.append(("⚠️", "Coke rate is high. Try increasing PCI rate to substitute coke."))
        if blast_temp < 1000:
            recs.append(("⚠️", "Blast temperature below 1000 °C. Increasing it improves hot metal output."))
        if moisture > 3.5:
            recs.append(("⚠️", "Moisture is high. Pre-drying raw materials saves thermal energy."))
        if oxygen_enrichment < 3.0:
            recs.append(("⚠️", "Oxygen enrichment is low. Increasing it improves thermal efficiency."))
        if pci_rate < 130 and coke_rate > 430:
            recs.append(("💡", "Low PCI + high coke: increase PCI to reduce coke dependency."))
        if efficiency < 55:
            recs.append(("⚠️", "Thermal efficiency well below average. Check blast temp & moisture."))
        if hot_metal > AVG_HOT_METAL + 50:
            recs.append(("✅", "Hot metal output is above average. Settings are performing well."))

        if not recs:
            st.success("✅ Parameters are in a reasonable operating range.")
        else:
            for icon, msg in recs:
                if icon == "✅":   st.success(f"{icon} {msg}")
                elif icon == "💡": st.info(f"{icon} {msg}")
                else:              st.warning(f"{icon} {msg}")

        st.markdown("---")
        st.markdown("#### 🎯 Optimized Parameters (based on your current input)")
        st.caption(
            "Grid search over blast_temp × coke_rate × pci_rate (7×7×4 = 196 combos), "
            "fixing oxygen, air flow, burden and moisture at your current values. "
            "Composite score: hot_metal − 0.3×fuel + 0.5×efficiency."
        )
        opt_params, opt_pred = smart_optimize(current_params, best_model)

        oc1, oc2, oc3 = st.columns(3)
        colored_metric(oc1, "🏭 Optimised Hot Metal",  f"{opt_pred[0]:.1f} t/day",
                       opt_pred[0] - hot_metal,  "positive")
        colored_metric(oc2, "⛽ Optimised Fuel",        f"{opt_pred[1]:.1f} kg/ton",
                       opt_pred[1] - fuel,        "negative")
        colored_metric(oc3, "🌡️ Optimised Efficiency",  f"{opt_pred[2]:.1f} %",
                       opt_pred[2] - efficiency,  "positive")
        st.caption("↑ Delta shown vs your current input (not vs average)")

        param_labels = {
            'blast_temp':        'Blast Temperature (°C)',
            'coke_rate':         'Coke Rate (kg/ton)',
            'pci_rate':          'PCI Rate (kg/ton)',
            'oxygen_enrichment': 'Oxygen Enrichment (%)',
            'air_flow':          'Air Flow (m³/min)',
            'burden_iron_ore':   'Burden Iron Ore (%)',
            'moisture':          'Moisture (%)',
        }
        rows = []
        for k, lbl in param_labels.items():
            cur   = current_params[k]
            opt   = opt_params[k]
            diff  = opt - cur
            arrow = "↑" if diff > 0.01 else ("↓" if diff < -0.01 else "→")
            rows.append({"Parameter": lbl,
                         "Your Value": f"{cur:.1f}",
                         "Suggested":  f"{opt:.1f}",
                         "Change":     f"{arrow} {abs(diff):.1f}"})
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True
        )
        st.caption("⚠️ Treat as directional guidance — based on synthetic data.")


# ──────────────────────────────────────────
# TAB 2 — Model Comparison
# ──────────────────────────────────────────
with tab2:
    st.subheader("Model Comparison — All 6 Algorithms")
    st.markdown("Trained and compared **6 ML algorithms** on an 800/200 train-test split.")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### R² Scores")
        st.dataframe(
            results.style
            .highlight_max(subset=['Average R²'], color='#d4f5e0')
            .highlight_min(subset=['Average RMSE'], color='#d4f5e0')
            .format({'Average R²': '{:.4f}', 'Average RMSE': '{:.4f}'}),
            width="stretch"
        )
        st.markdown("""
        **Key Finding:** Linear Regression (R² = 0.8443) outperformed all ensemble
        and boosting models. The dataset has predominantly **linear relationships** —
        consistent with the physics-driven nature of blast furnace operations.
        """)

    with col_b:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results))]
        bars   = ax.barh(results['Model'], results['Average R²'],
                         color=colors, edgecolor='#1a1a2e')
        ax.set_xlabel('Average R² Score', color='white')
        ax.set_title('Model Comparison', fontweight='bold', color='white')
        ax.set_xlim(0, 1.0)
        ax.invert_yaxis()
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#2d3561')
        for bar, val in zip(bars, results['Average R²']):
            ax.text(bar.get_width() + 0.008,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9, color='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("#### Actual vs Predicted — Linear Regression")
    targets       = ['hot_metal_output', 'fuel_consumption', 'thermal_efficiency']
    scatter_colors = ['#e74c3c', '#3498db', '#2ecc71']
    fig2, axes2   = plt.subplots(1, 3, figsize=(15, 5))
    fig2.patch.set_facecolor('#0e1117')

    for i, (target, color) in enumerate(zip(targets, scatter_colors)):
        actual    = y_test.iloc[:, i].values
        predicted = y_pred_lr[:, i]
        r2_val    = r2_score(actual, predicted)
        ax2       = axes2[i]
        ax2.set_facecolor('#0e1117')
        ax2.scatter(actual, predicted, alpha=0.4, color=color, s=20)
        mn = min(actual.min(), predicted.min())
        mx = max(actual.max(), predicted.max())
        ax2.plot([mn, mx], [mn, mx], 'w--', linewidth=1.5, label='Perfect fit')
        ax2.set_xlabel(f'Actual {target}',    fontsize=9,  color='white')
        ax2.set_ylabel(f'Predicted {target}', fontsize=9,  color='white')
        ax2.set_title(f'{target}\nR² = {r2_val:.4f}',
                      fontsize=10, fontweight='bold', color='white')
        ax2.legend(fontsize=8, labelcolor='white')
        ax2.tick_params(colors='white')
        ax2.spines[:].set_color('#2d3561')

    fig2.suptitle('Actual vs Predicted — Linear Regression',
                  fontsize=13, fontweight='bold', color='white')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")
    st.markdown("#### Per-Target R² Breakdown")
    st.markdown("""
    | Target | Linear R² | Interpretation |
    |---|---|---|
    | hot_metal_output   | 0.8114 | Strong — blast temp and coke rate dominate directly |
    | fuel_consumption   | 0.9239 | Excellent — coke + PCI are direct linear drivers of fuel |
    | thermal_efficiency | 0.7977 | Good — blast temp, oxygen, moisture all contribute linearly |
    """)


# ──────────────────────────────────────────
# TAB 3 — Feature Importance
# ──────────────────────────────────────────
with tab3:
    st.subheader("Feature Importance — Random Forest")
    st.markdown("Which input parameters have the most influence on predictions?")

    sorted_idx      = np.argsort(feat_imp)[::-1]
    sorted_features = [feature_cols[i] for i in sorted_idx]
    sorted_imp      = feat_imp[sorted_idx]

    col_p, col_q = st.columns([1, 1])
    with col_p:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        bar_colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db','#9b59b6','#95a5a6']
        bars = ax.barh(sorted_features, sorted_imp,
                       color=bar_colors[:len(sorted_features)], edgecolor='#1a1a2e')
        ax.set_xlabel('Importance Score', color='white')
        ax.set_title('Feature Importance — Random Forest',
                     fontweight='bold', color='white')
        ax.invert_yaxis()
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#2d3561')
        for bar, val in zip(bars, sorted_imp):
            ax.text(bar.get_width() + 0.003,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9, color='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_q:
        st.markdown("#### Why These Rankings Make Industrial Sense")
        st.markdown(f"""
        **coke_rate — {sorted_imp[0]*100:.1f}%** (dominant)
        Coke is the primary fuel AND the main reducing agent. Controls heat generation,
        chemical reduction, and burden permeability simultaneously.

        **blast_temp — {sorted_imp[1]*100:.1f}%** (second)
        Controls combustion intensity and the rate of iron ore reduction.
        The operator's primary real-time control lever.

        **pci_rate — {sorted_imp[2]*100:.1f}%** (third)
        Pulverized coal injection substitutes expensive coke (~0.8–0.9 kg per kg PCI),
        directly affecting fuel and output balance.

        **air_flow, moisture, oxygen_enrichment, burden_iron_ore — secondary**
        Important but less dominant than primary fuel and temperature controls.
        """)


# ──────────────────────────────────────────
# TAB 4 — About Project
# ──────────────────────────────────────────
with tab4:
    st.subheader("About This Project")
    st.markdown("""
    ### 🏭 Context
    Developed as part of a **4-week Vocational Training at SAIL Bokaro Steel Plant**
    (April–May 2026), under the Control & Automation (C&A) Department.

    ---

    ### 📊 Dataset
    - **Type:** Synthetic, generated using domain-realistic parameter ranges
    - **Size:** 1000 rows × 10 columns (7 inputs, 3 outputs)
    - **Split:** 80% train / 20% test (random_state = 42)
    - **Why synthetic:** Real SAIL operational data is not publicly available.

    **Output column statistics (from generated data):**
    | Column | Mean | Std | Min | Max |
    |---|---|---|---|---|
    | hot_metal_output (t/day) | 2791.40 | 48.23 | 2659.94 | 2921.97 |
    | fuel_consumption (kg/ton) | 478.38 | 57.82 | 348.39 | 619.76 |
    | thermal_efficiency (%) | 60.76 | 6.66 | 41.65 | 82.59 |

    ---

    ### 🤖 Models Compared
    | Model | Avg R² | Avg RMSE |
    |---|---|---|
    | **Linear Regression** 🏆 | **0.8443** | **13.57** |
    | Polynomial Regression | 0.8362 | 13.83 |
    | XGBoost | 0.8218 | 14.50 |
    | Gradient Boosting | 0.8216 | 14.63 |
    | Random Forest | 0.7840 | 15.07 |
    | Extra Trees | 0.7791 | 15.20 |

    ---

    ### 📉 Key Finding
    **Linear Regression performed best** (Average R² = 0.8443). The blast furnace
    dataset has strong linear relationships by design — consistent with the
    physics-driven nature of industrial steelmaking processes.

    ---

    ### ⚠️ Known Limitations
    - Synthetic data — real deployment needs actual plant sensor data
    - No temporal modeling — real furnaces have time-delay effects
    - Missing variables: slag chemistry, furnace pressure, coke quality, tapping intervals
    - Optimization covers 3 of 7 parameters in the grid; others fixed at current input

    ---

    ### 👩‍💻 Built By
    **Aastha Jaiswal** | B.Tech CSE, KIIT University 2026

    *SAIL Bokaro Steel Plant Internship | BSL URN: 5914843*
    """)


st.markdown("---")
st.caption("Blast Furnace AI Optimization System | Aastha Jaiswal | KIIT 2026 | SAIL Bokaro")