import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ==========================================
# 1. Page Configuration & Dark-Mode Styling
# ==========================================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        color: #ffffff;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #6366f1;
        margin-bottom: 15px;
    }
    .persona-badge {
        font-size: 1.35rem;
        font-weight: 700;
    }
    .metric-card p {
        color: #e2e8f0;
        margin: 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Model & Scaler Loader (Handles 50k - 100k+ rows)
# ==========================================
@st.cache_resource
def load_data_and_models():
    """Loads dataset, scaler, and trained KMeans model."""
    data_path = "Mall_Customers.csv"
    model_path = "kmeans_model.pkl"
    scaler_path = "scaler.pkl"

    if not os.path.exists(data_path):
        st.error("Mall_Customers.csv not found in project folder!")
        st.stop()

    df = pd.read_csv(data_path)

    # Standardize column names if needed
    col_map = {}
    for col in df.columns:
        c_lower = col.lower().replace("_", " ").strip()
        if "income" in c_lower:
            col_map[col] = "Annual Income (k$)"
        elif "spending" in c_lower or "score" in c_lower:
            col_map[col] = "Spending Score (1-100)"
        elif "customer" in c_lower and "id" in c_lower:
            col_map[col] = "CustomerID"
    
    if col_map:
        df = df.rename(columns=col_map)

    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    if not all(col in df.columns for col in features):
        st.error(f"Dataset must contain 'Annual Income (k$)' and 'Spending Score (1-100)'. Found: {list(df.columns)}")
        st.stop()

    X = df[features]

    # Load or fit scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(X)

    # Load or fit KMeans model
    if os.path.exists(model_path):
        kmeans = joblib.load(model_path)
    else:
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=5)
        kmeans.fit(scaler.transform(X))

    # Assign clusters
    df["Cluster"] = kmeans.predict(scaler.transform(X))
    return kmeans, scaler, df

kmeans, scaler, data = load_data_and_models()

# ==========================================
# 3. Dynamic Persona Mapping
# ==========================================
def get_personas(model, scaler_obj):
    """Maps numerical cluster IDs to real-world business marketing profiles."""
    centers_scaled = model.cluster_centers_
    centers_real = scaler_obj.inverse_transform(centers_scaled)

    persona_info = {}
    med_inc = np.median(centers_real[:, 0])
    med_sc = np.median(centers_real[:, 1])

    for idx, (inc, sc) in enumerate(centers_real):
        if inc > med_inc and sc > med_sc:
            name = "VIP / High Value"
            desc = "High Income & High Spending"
            rec = "Loyalty perks, VIP club access, exclusive previews, luxury product upsells."
            color = "#10B981"
        elif inc > med_inc and sc <= med_sc:
            name = "Savers / Cautious Affluent"
            desc = "High Income & Low Spending"
            rec = "Emphasize long-term value, durability, premium quality, investment deals."
            color = "#3B82F6"
        elif inc <= med_inc and sc > med_sc:
            name = "Trendsetters / Impulse Buyers"
            desc = "Moderate-to-Low Income & High Spending"
            rec = "Flash discounts, bundle offers, trending fashion, BNPL options."
            color = "#F59E0B"
        elif inc <= med_inc and sc <= med_sc and (inc < med_inc * 0.75 and sc < med_sc * 0.75):
            name = "Budget Shoppers"
            desc = "Low Income & Low Spending"
            rec = "Clearance alerts, promotional coupons, bulk-buy value packs."
            color = "#EF4444"
        else:
            name = "Balanced Shoppers"
            desc = "Average Income & Average Spending"
            rec = "Seasonal marketing, reward points, dependable customer support."
            color = "#8B5CF6"

        persona_info[idx] = {
            "name": name,
            "desc": desc,
            "recommendation": rec,
            "color": color,
            "income": round(inc, 1),
            "score": round(sc, 1)
        }
    return persona_info

personas = get_personas(kmeans, scaler)
data["Segment"] = data["Cluster"].map(lambda c: personas[c]["name"])

# ==========================================
# 4. Streamlit Multi-Tab User Interface
# ==========================================
st.title("🛒 Customer Segmentation & Behavioral Analytics")
st.caption("Machine Learning Application for Targeted Customer Personas using K-Means Clustering")

tab1, tab2, tab3 = st.tabs([
    "🎯 Live Predictor",
    "📊 Cluster Analytics",
    "📋 Data Explorer"
])

# ----------------- TAB 1: PREDICTOR -----------------
with tab1:
    col_input, col_result = st.columns([1, 1.4], gap="large")

    with col_input:
        st.subheader("Customer Metrics")
        st.write("Enter income and spending behavior to predict customer persona:")

        max_inc = int(data["Annual Income (k$)"].max())
        min_inc = int(data["Annual Income (k$)"].min())

        user_income = st.slider(
            "Annual Income (in $1,000s):",
            min_value=max(0, min_inc - 5),
            max_value=max(150, max_inc + 10),
            value=int(data["Annual Income (k$)"].median()),
            step=1
        )

        user_score = st.slider(
            "Spending Score (1 - 100):",
            min_value=1,
            max_value=100,
            value=50,
            step=1
        )

        input_df = pd.DataFrame({
            "Annual Income (k$)": [user_income],
            "Spending Score (1-100)": [user_score]
        })

        scaled_point = scaler.transform(input_df)
        pred_cluster = int(kmeans.predict(scaled_point)[0])
        persona = personas[pred_cluster]

    with col_result:
        st.subheader("Segment Classification")
        st.markdown(f"""
        <div class="metric-card">
            <span class="persona-badge" style="color: {persona['color']};">● {persona['name']}</span>
            <p style="margin-top: 8px;"><strong>Characteristics:</strong> {persona['desc']}</p>
            <p><strong>Cluster ID:</strong> #{pred_cluster}</p>
            <p><strong>Recommended Strategy:</strong> {persona['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)

        # High-performance visual sampling for large datasets (>5,000 rows)
        if len(data) > 4000:
            plot_df = data.sample(n=3000, random_state=42)
            plot_title = f"Cluster Space (Displaying 3,000 sampled points out of {len(data):,} customers)"
        else:
            plot_df = data
            plot_title = f"Cluster Space ({len(data):,} total customers)"

        fig = px.scatter(
            plot_df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color="Segment",
            color_discrete_map={info["name"]: info["color"] for info in personas.values()},
            title=plot_title,
            opacity=0.60
        )

        fig.add_trace(
            go.Scatter(
                x=[user_income],
                y=[user_score],
                mode="markers",
                name="Current Input",
                marker=dict(size=18, color="white", symbol="star", line=dict(width=2, color="gold"))
            )
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------------- TAB 2: OVERVIEW -----------------
with tab2:
    st.subheader("Cluster Personas Summary")
    card_cols = st.columns(len(personas))
    for i, (cid, p) in enumerate(personas.items()):
        with card_cols[i]:
            count = int((data["Cluster"] == cid).sum())
            pct = round((count / len(data)) * 100, 1)
            st.metric(label=p["name"], value=f"{count:,} users", delta=f"{pct}% of total")
            st.caption(f"Avg Inc: ${p['income']}k | Avg Score: {p['score']}")

    st.divider()

    st.subheader("Statistical Summary by Cluster")
    summary_table = data.groupby("Segment")[["Annual Income (k$)", "Spending Score (1-100)"]].agg(["mean", "min", "max", "count"]).round(1)
    st.dataframe(summary_table, use_container_width=True)

# ----------------- TAB 3: DATA -----------------
with tab3:
    st.subheader(f"Customer Records ({len(data):,} total rows)")
    display_cols = [c for c in ["CustomerID", "Annual Income (k$)", "Spending Score (1-100)", "Cluster", "Segment"] if c in data.columns]
    st.dataframe(data[display_cols].head(500), use_container_width=True)
    st.caption("Showing the first 500 records for fast browser rendering.")