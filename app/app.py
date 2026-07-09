"""
House Price Prediction App
Built by: Amna Arshad — AI Engineer
Model: Lasso Regression (tuned) | Dataset: Ames Housing (Kaggle)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="House Price Predictor | Amna Arshad",
    page_icon="🏠",
    layout="wide"
)

# ----------------------------
# Load Artifacts
# ----------------------------
# Build paths relative to THIS script's location, not the current working
# directory - Streamlit Cloud runs the app from the repo root, so plain
# filenames like 'lasso_model.pkl' would fail to resolve to app/lasso_model.pkl
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def _path(filename):
    return os.path.join(APP_DIR, filename)

@st.cache_resource
def load_artifacts():
    model = joblib.load(_path('lasso_model.pkl'))
    scaler = joblib.load(_path('scaler.pkl'))
    feature_columns = joblib.load(_path('feature_columns.pkl'))
    default_row = joblib.load(_path('default_row.pkl'))
    neighborhood_options = joblib.load(_path('neighborhood_options.pkl'))
    feature_stats = joblib.load(_path('feature_stats.pkl'))
    train_price_bounds = joblib.load(_path('train_price_bounds.pkl'))
    return model, scaler, feature_columns, default_row, neighborhood_options, feature_stats, train_price_bounds

try:
    model, scaler, feature_columns, default_row, neighborhood_options, feature_stats, train_price_bounds = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError as e:
    artifacts_loaded = False
    missing_file = str(e)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #fafbff 0%, #ffffff 100%);
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.02rem;
        color: #6b7280;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #4f46e5;
        background-color: #f0f0ff;
        padding: 8px 14px;
        border-radius: 8px;
        margin-bottom: 14px;
        display: inline-block;
    }
    .price-box {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 60%, #a855f7 100%);
        padding: 36px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 24px 0;
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.25);
    }
    .price-box h1 {
        font-size: 3.2rem;
        margin: 6px 0;
        color: white;
        font-weight: 800;
    }
    .price-box p {
        font-size: 1rem;
        opacity: 0.92;
        margin: 2px 0;
    }
    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #ececff;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    .metric-card h4 {
        color: #7c3aed;
        margin-bottom: 6px;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card p {
        color: #1f2937;
        font-weight: 700;
        font-size: 1.05rem;
        margin: 0;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        padding: 12px 0;
        font-size: 1.05rem;
        box-shadow: 0 6px 16px rgba(124, 58, 237, 0.3);
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(124, 58, 237, 0.4);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4f46e5 0%, #6d28d9 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #f5f3ff !important;
    }
    section[data-testid="stSidebar"] .stMetric {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 8px;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2);
    }
    .sidebar-card {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 16px;
        border: 1px solid rgba(255,255,255,0.15);
    }
    .sidebar-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        margin: 3px 4px 3px 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown('<p class="main-title">🏠 House Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered price estimation using Lasso Regression | Trained on Ames Housing Dataset (Kaggle)</p>', unsafe_allow_html=True)
st.markdown("---")

if not artifacts_loaded:
    st.error(f"""
    ⚠️ **Model files not found.**
    
    Please run the notebook cells that save the following files in the same folder as `app.py`:
    - lasso_model.pkl
    - scaler.pkl
    - feature_columns.pkl
    - default_row.pkl
    - neighborhood_options.pkl
    - feature_stats.pkl
    - train_price_bounds.pkl
    
    Missing: {missing_file}
    """)
    st.stop()

# ----------------------------
# Sidebar - About
# ----------------------------
with st.sidebar:
    st.markdown("## 🏠 House Price AI")
    st.markdown(
        '<div class="sidebar-card">'
        '📈 Predicts home sale prices using a <b>tuned Lasso Regression</b> model, '
        'trained on the Ames Housing dataset (1,460 homes, 193 engineered features).'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### ⚙️ Pipeline")
    st.markdown(
        '<div class="sidebar-card">'
        '<span class="sidebar-badge">🧹 Missing Value Imputation</span>'
        '<span class="sidebar-badge">📏 IQR Outlier Removal</span>'
        '<span class="sidebar-badge">📉 Log Transform</span>'
        '<span class="sidebar-badge">🔢 Ordinal + One-Hot Encoding</span>'
        '<span class="sidebar-badge">⚖️ StandardScaler</span>'
        '<span class="sidebar-badge">🎯 GridSearchCV Tuning</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🏆 Model Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Val R²", "0.9096")
    with c2:
        st.metric("Val RMSE", "0.1235")

    st.markdown(
        '<div class="sidebar-card">'
        '<b>Models Compared:</b><br>'
        'Linear Regression → R² 0.898<br>'
        'Ridge Regression → R² 0.906<br>'
        '<b>Lasso Regression → R² 0.910 ✅</b><br><br>'
        'Lasso auto-selected <b>65 / 193</b> most important features.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; font-size:0.85rem; opacity:0.85;">'
        'Built with 💜 by <b>Amna Arshad</b><br>AI Engineer</p>',
        unsafe_allow_html=True
    )

# ----------------------------
# Input Form
# ----------------------------
st.subheader("Enter House Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="section-header">🏗️ Structure</p>', unsafe_allow_html=True)
    overall_qual = st.slider("Overall Quality (1=Poor, 10=Excellent)",
                               1, 10, feature_stats['OverallQual'][2])
    year_built = st.slider("Year Built",
                             feature_stats['YearBuilt'][0], feature_stats['YearBuilt'][1], feature_stats['YearBuilt'][2])
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)",
                                    min_value=200, max_value=6000,
                                    value=feature_stats['GrLivArea'][2], step=50)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)",
                                      min_value=0, max_value=6000,
                                      value=feature_stats['TotalBsmtSF'][2], step=50)

with col2:
    st.markdown('<p class="section-header">🛏️ Rooms</p>', unsafe_allow_html=True)
    full_bath = st.slider("Full Bathrooms", 0, 4, feature_stats['FullBath'][2])
    bedroom_abv_gr = st.slider("Bedrooms (above ground)", 0, 8, feature_stats['BedroomAbvGr'][2])
    garage_cars = st.slider("Garage Capacity (cars)", 0, 4, feature_stats['GarageCars'][2])
    lot_area = st.number_input("Lot Area (sq ft)",
                                 min_value=1000, max_value=50000,
                                 value=feature_stats['LotArea'][2], step=500)

with col3:
    st.markdown('<p class="section-header">📍 Location</p>', unsafe_allow_html=True)
    neighborhood = st.selectbox("Neighborhood", neighborhood_options)
    central_air = st.radio("Central Air Conditioning", ["Yes", "No"], horizontal=True)
    kitchen_qual = st.select_slider("Kitchen Quality",
                                      options=["Fair", "Typical", "Good", "Excellent"],
                                      value="Typical")

with st.expander("⚙️ Advanced Options (optional)"):
    a1, a2, a3 = st.columns(3)
    with a1:
        fireplaces = st.slider("Fireplaces", 0, 3, 1)
    with a2:
        has_pool = st.checkbox("Has Pool")
    with a3:
        remod_recent = st.checkbox("Recently Remodeled")

st.markdown("---")
predict_btn = st.button("🔮 Predict House Price", use_container_width=True, type="primary")

# ----------------------------
# Prediction Logic
# ----------------------------
if predict_btn:
    # Start from the default (median) row so every one of the 193 features has a sensible value
    input_row = default_row.copy()

    # Override numeric features the user actually controls
    input_row['OverallQual'] = overall_qual
    input_row['YearBuilt'] = year_built
    input_row['GrLivArea'] = gr_liv_area
    input_row['TotalBsmtSF'] = total_bsmt_sf
    input_row['FullBath'] = full_bath
    input_row['BedroomAbvGr'] = bedroom_abv_gr
    input_row['GarageCars'] = garage_cars
    input_row['LotArea'] = lot_area

    # Central Air (binary-encoded column from one-hot: CentralAir_Y)
    if 'CentralAir_Y' in input_row.index:
        input_row['CentralAir_Y'] = 1 if central_air == "Yes" else 0

    # Kitchen Quality (ordinal-encoded: 1=Fa, 3=TA, 4=Gd, 5=Ex from our mapping)
    qual_map = {"Fair": 2, "Typical": 3, "Good": 4, "Excellent": 5}
    if 'KitchenQual' in input_row.index:
        input_row['KitchenQual'] = qual_map[kitchen_qual]

    # Advanced options
    if 'Fireplaces' in input_row.index:
        input_row['Fireplaces'] = fireplaces
    if 'PoolArea' in input_row.index:
        input_row['PoolArea'] = 500 if has_pool else 0
    if 'YearRemodAdd' in input_row.index:
        # Use a recent-but-realistic year within training data bounds (dataset max ~2010)
        input_row['YearRemodAdd'] = max(year_built, 2009) if remod_recent else year_built

    # Neighborhood (one-hot: reset all Neighborhood_* to 0, then set selected one to 1)
    for col in input_row.index:
        if col.startswith('Neighborhood_'):
            input_row[col] = 0
    selected_col = f'Neighborhood_{neighborhood}'
    if selected_col in input_row.index:
        input_row[selected_col] = 1
    # Note: if neighborhood matches the baseline/dropped category, all dummies stay 0 (correct)

    # Build final input dataframe in correct column order
    input_df = pd.DataFrame([input_row])[feature_columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict (log scale) -> convert back
    pred_log = model.predict(input_scaled)[0]
    pred_price = np.expm1(pred_log)

    # Cap within realistic bounds (same logic as test predictions)
    lower_bound = train_price_bounds[0] * 0.7
    upper_bound = train_price_bounds[1] * 1.3
    pred_price_capped = float(np.clip(pred_price, lower_bound, upper_bound))

    # ----------------------------
    # Display Result
    # ----------------------------
    st.markdown(f"""
    <div class="price-box">
        <p>Estimated Sale Price</p>
        <h1>${pred_price_capped:,.0f}</h1>
        <p>Based on the features you provided</p>
    </div>
    """, unsafe_allow_html=True)

    if pred_price != pred_price_capped:
        st.info("ℹ️ Note: The raw prediction was adjusted to stay within a realistic range based on the training data distribution.")

    # ----------------------------
    # Feature Importance (Lasso coefficients)
    # ----------------------------
    st.subheader("📈 What's Driving This Prediction?")

    coefs = pd.Series(model.coef_, index=feature_columns)
    nonzero_coefs = coefs[coefs != 0]
    top_features = nonzero_coefs.abs().sort_values(ascending=False).head(10)
    top_features_signed = coefs[top_features.index]

    fig = go.Figure(go.Bar(
        x=top_features_signed.values,
        y=top_features_signed.index,
        orientation='h',
        marker_color=['#22c55e' if v > 0 else '#ef4444' for v in top_features_signed.values]
    ))
    fig.update_layout(
        title="Top 10 Most Influential Features (Lasso Coefficients)",
        xaxis_title="Impact on Price (log scale)",
        yaxis=dict(autorange="reversed"),
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 Green = increases price | 🔴 Red = decreases price")

    # ----------------------------
    # Model Stats Row
    # ----------------------------
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><h4>Model</h4><p>Lasso Regression (α=0.005)</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h4>Validation R²</h4><p>0.9096</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h4>Features Used</h4><p>65 / 193 (Lasso-selected)</p></div>', unsafe_allow_html=True)

else:
    st.info("👆 Adjust the house details above and click **Predict House Price** to get an estimate.")
