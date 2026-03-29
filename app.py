"""
app.py  —  AI Skill Suggester
"What Can I Learn Today?"
------------------------------
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Try TensorFlow ─────────────────────────────────────────────────────────────
try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="What Can I Learn Today?",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background - deep charcoal with subtle purple tint */
    .stApp {
        background-color: #13111a;
    }

    /* Sidebar - gradient dark purple */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1525 0%, #1c1428 100%) !important;
        border-right: 1px solid #2d2440;
    }

    [data-testid="stSidebar"] * {
        color: #cccccc !important;
    }

    /* Headers */
    h1, h2, h3 { font-family: 'DM Sans', sans-serif !important; color: #eeeeee !important; }

    /* Hero - vibrant gradient */
    .hero-container {
        background: linear-gradient(135deg, #1e1535 0%, #1a2235 50%, #1e2a20 100%);
        border: 1px solid #3d2f6a;
        border-radius: 20px;
        padding: 2.8rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -60px; left: -60px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, #7c3aed22 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -40px; right: -40px;
        width: 180px; height: 180px;
        background: radial-gradient(circle, #059669220%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 2.7rem;
        font-weight: 700;
        color: #f0eeff;
        margin: 0;
        line-height: 1.2;
    }
    .hero-title span {
        background: linear-gradient(90deg, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #9988bb;
        margin-top: 0.7rem;
    }

    /* Cards */
    .result-card {
        background: linear-gradient(135deg, #1e1535 0%, #1a1f30 100%);
        border: 1px solid #3d2f6a;
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 24px #7c3aed12;
    }
    .result-card-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #f0eeff;
        margin: 0 0 0.3rem 0;
    }
    .result-card-meta {
        font-size: 0.85rem;
        color: #8877aa;
        margin-bottom: 0.6rem;
    }
    .result-card-desc {
        font-size: 0.93rem;
        color: #b8a8d0;
        line-height: 1.65;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.22rem 0.75rem;
        border-radius: 50px;
        font-size: 0.76rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .badge-beginner    { background: #0d2e1a; color: #34d399; border: 1px solid #065f3a; }
    .badge-intermediate{ background: #2e1f06; color: #fbbf24; border: 1px solid #6f4b10; }
    .badge-advanced    { background: #2e0d1a; color: #f87171; border: 1px solid #6f1030; }
    .badge-free        { background: #0d2e1a; color: #34d399; border: 1px solid #065f3a; }
    .badge-paid        { background: #1a1040; color: #a78bfa; border: 1px solid #4c2fb0; }

    /* Stat boxes */
    .stat-box {
        background: linear-gradient(135deg, #1e1535 0%, #1a1f30 100%);
        border: 1px solid #3d2f6a;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        box-shadow: 0 4px 20px #7c3aed0e;
        transition: transform 0.2s;
    }
    .stat-box:hover { transform: translateY(-2px); }
    .stat-number {
        font-family: 'DM Sans', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #8877aa;
        margin-top: 0.25rem;
    }

    /* Section headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 0.68rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2d2050;
    }

    /* Input labels */
    label { color: #9988bb !important; font-size: 0.88rem !important; }

    /* Selectbox and multiselect */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #1e1535 !important;
        border: 1px solid #3d2f6a !important;
        color: #cccccc !important;
        border-radius: 10px !important;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #059669) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4c1d95, #065f46) !important;
        color: #ffffff !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: 1px solid #7c3aed66 !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        width: 100% !important;
        transition: all 0.25s !important;
        box-shadow: 0 4px 16px #7c3aed22 !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6d28d9, #047857) !important;
        border-color: #a78bfa !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px #7c3aed44 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #2d2050;
        gap: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #776699 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.5rem 1.2rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1e1535 !important;
        color: #a78bfa !important;
        border-bottom: 2px solid #7c3aed !important;
    }

    /* Dividers */
    hr { border-color: #2d2050 !important; }

    /* Matplotlib transparent bg */
    .stPlotlyChart, [data-testid="stImage"] { border-radius: 12px; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Pulse animation for the main button */
    @keyframes pulse-glow {
        0%   { box-shadow: 0 0 0 0 #7c3aed44; }
        70%  { box-shadow: 0 0 0 10px #7c3aed00; }
        100% { box-shadow: 0 0 0 0 #7c3aed00; }
    }
    .stButton > button {
        animation: pulse-glow 2.5s infinite;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS & DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_dataset():
    return pd.read_csv("skills_dataset.csv")

@st.cache_resource
def load_nn():
    if TF_AVAILABLE:
        try:
            return keras.models.load_model("nn_model.keras")
        except:
            return None
    return None

try:
    artifacts = load_artifacts()
    df = load_dataset()
    nn_model = load_nn()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def encode_input(user_difficulty, user_cost, user_hours, user_interests, artifacts):
    """Convert user inputs into the feature vector the model expects."""
    difficulty_val = artifacts["difficulty_map"].get(user_difficulty, 1)
    cost_val = artifacts["cost_map"].get(user_cost, 0)

    # Build tag vector
    all_tags = artifacts["mlb"].classes_
    tag_vector = np.array([
        1 if tag in [t.strip() for interest in user_interests
                     for t in interest.lower().replace(" ", "").split("/")]
        else 0
        for tag in all_tags
    ])

    feature_vec = np.array([[difficulty_val, cost_val, user_hours] + list(tag_vector)])
    return feature_vec


def get_skill_details(skill_name, df):
    row = df[df["skill_name"] == skill_name]
    if row.empty:
        return None
    return row.iloc[0]


def difficulty_badge(diff):
    cls = {"Beginner": "beginner", "Intermediate": "intermediate", "Advanced": "advanced"}.get(diff, "beginner")
    return f'<span class="badge badge-{cls}">{diff}</span>'


def cost_badge(cost):
    cls = "free" if cost == "Free" else "paid"
    return f'<span class="badge badge-{cls}">{cost}</span>'


def make_radar_chart(skill_row):
    """Radar chart showing skill profile."""
    categories = ["Difficulty", "Time\nCommitment", "Cost", "Demand", "Accessibility"]
    diff_map = {"Beginner": 2, "Intermediate": 3.5, "Advanced": 5}
    cost_map_val = {"Free": 1, "Paid": 3.5}

    hours = float(skill_row["weekly_hours_needed"])
    values = [
        diff_map.get(skill_row["difficulty"], 2),
        min(hours / 2, 5),
        cost_map_val.get(skill_row["cost_level"], 1),
        4.2,
        5 - cost_map_val.get(skill_row["cost_level"], 1)
    ]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#13111a")
    ax.set_facecolor("#13111a")

    ax.plot(angles, values, color="#a78bfa", linewidth=2)
    ax.fill(angles, values, color="#7c3aed", alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="#9988bb", size=8)
    ax.set_yticklabels([])
    ax.set_ylim(0, 5)
    ax.spines['polar'].set_color("#3d2f6a")
    ax.grid(color="#3d2f6a", linestyle="--", linewidth=0.7)

    return fig


def make_category_chart(df):
    """Bar chart: skills per category."""
    counts = df["category"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#13111a")
    ax.set_facecolor("#13111a")

    bar_colors = ["#7c3aed", "#6d28d9", "#5b21b6", "#a78bfa", "#34d399", "#059669", "#065f46", "#047857"]
    bars = ax.barh(counts.index, counts.values,
                   color=[bar_colors[i % len(bar_colors)] for i in range(len(counts))],
                   alpha=0.9, height=0.6)
    ax.set_xlabel("Number of Skills", color="#9988bb", fontsize=9)
    ax.tick_params(colors="#9988bb", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d2050")
    ax.xaxis.set_tick_params(color="#88888820")

    for bar, val in zip(bars, counts.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color="#f0eeff", fontsize=8, fontweight="bold")

    plt.tight_layout()
    return fig


def make_difficulty_pie(df):
    """Donut chart: difficulty distribution."""
    counts = df["difficulty"].value_counts()
    colors = ["#34d399", "#fbbf24", "#f87171"]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor("#13111a")
    ax.set_facecolor("#13111a")

    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, colors=colors,
        autopct="%1.0f%%", pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor="#1a1a1a", linewidth=2)
    )
    for t in texts:
        t.set_color("#b8a8d0")
        t.set_fontsize(9)
    for at in autotexts:
        at.set_color("#1a1a1a")
        at.set_fontsize(8)
        at.set_fontweight("bold")

    plt.tight_layout()
    return fig


def make_hours_chart(df):
    """Hours needed distribution."""
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#13111a")
    ax.set_facecolor("#13111a")

    diff_colors = {"Beginner": "#34d399", "Intermediate": "#fbbf24", "Advanced": "#f87171"}
    for diff, grp in df.groupby("difficulty"):
        ax.scatter(grp["skill_name"], grp["weekly_hours_needed"],
                   color=diff_colors[diff], label=diff, s=60, alpha=0.85, zorder=3)

    ax.set_ylabel("Weekly Hours", color="#9988bb", fontsize=9)
    ax.tick_params(colors="#9988bb", labelsize=7, axis="both")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", color="#2d205040", linestyle="--")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d2050")

    legend = ax.legend(fontsize=8, facecolor="#1e1535", edgecolor="#3d2f6a", labelcolor="#b8a8d0")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-size:2.8rem; filter: drop-shadow(0 0 12px #7c3aed88);'>🧠</div>
        <div style='font-family: DM Sans, sans-serif; font-size:1.15rem; font-weight:800;
                    background: linear-gradient(90deg, #a78bfa, #34d399);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
                    margin-top:0.4rem;'>Skill Suggester</div>
        <div style='font-size:0.75rem; color:#776699; margin-top:0.2rem;'>✨ AI-Powered Learning Guide</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">⚙️ YOUR PROFILE</div>', unsafe_allow_html=True)

    user_name = st.text_input("Your name", placeholder="e.g. Sophie")

    current_skills = st.multiselect(
        "Skills you already have",
        options=sorted(df["skill_name"].tolist()) if models_loaded else [],
        placeholder="Select your current skills..."
    )

    interests = st.multiselect(
        "Your interests",
        options=["tech", "data", "design", "web", "coding", "AI", "business",
                 "marketing", "writing", "creative", "finance", "life", "career",
                 "leadership", "analytics", "communication", "IT", "mobile", "media"],
        placeholder="What do you enjoy?"
    )

    st.markdown('<div class="section-header" style="margin-top:1rem;">⏱️ YOUR CONSTRAINTS</div>', unsafe_allow_html=True)

    weekly_hours = st.slider("Hours available per week", 1, 15, 5)

    budget = st.selectbox("Budget for learning", ["Free only", "Willing to pay"])
    budget_val = "Free" if budget == "Free only" else "Paid"

    difficulty_pref = st.selectbox(
        "Preferred difficulty level",
        ["Beginner", "Intermediate", "Advanced"]
    )

    # Always use best available model silently
    use_nn = nn_model is not None

    st.markdown("---")
    suggest_btn = st.button("✨ Suggest My Next Skill", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
if not models_loaded:
    st.error("""
    ⚠️ **Models not found.**
    Please run the training script first:
    ```
    python train_model.py
    ```
    Then relaunch the app.
    """)
    st.stop()

# Hero
greeting = f"Hello, {user_name}! " if user_name else ""
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">{greeting}<span>"What Can I Learn Today?"</span></div>
    <div class="hero-subtitle">Enter your profile on the left → Get a personalized skill recommendation powered by AI</div>
</div>
""", unsafe_allow_html=True)


# Stats row
c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="stat-box"><div class="stat-number" style="color:#a78bfa;">59%</div><div class="stat-label">⚡ Workforce Must Reskill by 2030</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-box"><div class="stat-number" style="color:#34d399;">1B+</div><div class="stat-label">🌍 People Need to Upskill Globally</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RECOMMENDATION SECTION
# ─────────────────────────────────────────────────────────────
if True:
    if suggest_btn:
        if not interests:
            st.warning("Please select at least one interest from the sidebar.")
        else:
            with st.spinner("Finding your best match..."):
                feature_vec = encode_input(difficulty_pref, budget_val, weekly_hours, interests, artifacts)
                rf_model = artifacts["rf_model"]
                le_skill = artifacts["le_skill"]

                if use_nn and nn_model is not None:
                    probs = nn_model.predict(feature_vec, verbose=0)[0]
                else:
                    probs = rf_model.predict_proba(feature_vec)[0]

                top_indices = np.argsort(probs)[::-1][:5]
                top_skills = [(le_skill.classes_[i], probs[i]) for i in top_indices]
                top_skills = [(s, p) for s, p in top_skills if s not in current_skills]

            if not top_skills:
                st.info("You seem to know all the top recommendations! Try adjusting your profile.")
            else:
                primary_skill, primary_conf = top_skills[0]
                primary_details = get_skill_details(primary_skill, df)

                # ── Primary Recommendation Card ──
                st.markdown('<div class="section-header">TOP RECOMMENDATION</div>', unsafe_allow_html=True)
                col_main, col_chart = st.columns([3, 1])
                with col_main:
                    if primary_details is not None:
                        badges = difficulty_badge(primary_details["difficulty"]) + cost_badge(primary_details["cost_level"])
                        quick_links = ""
                        for r in range(1, 4):
                            rname = primary_details.get(f"resource_{r}_name", "")
                            rurl = primary_details.get(f"resource_{r}_url", "")
                            if rname and rurl and str(rname) != "nan":
                                quick_links += f'<a href="{rurl}" target="_blank" style="display:inline-block; margin-right:0.5rem; margin-top:0.4rem; padding:0.2rem 0.7rem; background:#2e2e2e; border:1px solid #444444; border-radius:50px; color:#cccccc; font-size:0.78rem; text-decoration:none;">🔗 {rname}</a>'
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-card-title">📚 {primary_skill}</div>
                            <div class="result-card-meta">
                                {badges}
                                <span style="color:#888888; font-size:0.82rem;">
                                    &nbsp;⏱ {primary_details['weekly_hours_needed']}h/week &nbsp;·&nbsp;
                                    🖥 {primary_details['platform']} &nbsp;·&nbsp;
                                    📂 {primary_details['category']}
                                </span>
                            </div>
                            <div class="result-card-desc">{primary_details['description']}</div>
                            <div style="margin-top:0.8rem;">{quick_links}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        conf_pct = int(primary_conf * 100)
                        st.markdown(f"""
                        <div style="margin-top:0.5rem;">
                            <div style="font-size:0.8rem; color:#9988bb; margin-bottom:0.3rem;">
                                Match confidence: <strong style="color:#a78bfa;">{conf_pct}%</strong>
                            </div>
                            <div style="background:#1e1535; border-radius:6px; height:7px; overflow:hidden; border:1px solid #3d2f6a;">
                                <div style="width:{conf_pct}%; background:linear-gradient(90deg,#7c3aed,#34d399); height:100%; border-radius:6px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                with col_chart:
                    if primary_details is not None:
                        fig = make_radar_chart(primary_details)
                        st.pyplot(fig, use_container_width=True)
                        plt.close()

                # ── Learning Path ──
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">YOUR LEARNING PATH</div>', unsafe_allow_html=True)
                prereq = str(primary_details["prerequisite_skills"]) if primary_details is not None else "None"
                if prereq and prereq.lower() != "none":
                    prereq_known = prereq in current_skills
                    prereq_status = "You already know this!" if prereq_known else "Learn this first"
                    prereq_color = "#7ec98a" if prereq_known else "#c9a96e"
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:1rem; padding:0.8rem 1rem;
                                background:#252525; border:1px solid #333333; border-radius:10px; margin-bottom:0.8rem;">
                        <span style="font-size:1.2rem;">📌</span>
                        <div>
                            <div style="font-size:0.72rem; color:#666666; letter-spacing:1px; text-transform:uppercase;">Prerequisite</div>
                            <div style="color:{prereq_color}; font-weight:600;">{prereq} — {prereq_status}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:1rem; padding:0.8rem 1rem;
                            background:#2a2a2a; border:1px solid #3a3a3a; border-radius:10px;">
                    <span style="font-size:1.2rem;">🎯</span>
                    <div>
                        <div style="font-size:0.72rem; color:#666666; letter-spacing:1px; text-transform:uppercase;">Learn Now</div>
                        <div style="color:#eeeeee; font-weight:700; font-size:1.05rem;">{primary_skill}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Learning Resources ──
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">LEARNING RESOURCES</div>', unsafe_allow_html=True)
                if primary_details is not None:
                    res_cols = st.columns(3)
                    icons = ["🎓", "📘", "▶️"]
                    for r in range(1, 4):
                        rname = primary_details.get(f"resource_{r}_name", "")
                        rurl = primary_details.get(f"resource_{r}_url", "")
                        if rname and rurl and str(rname) != "nan":
                            with res_cols[r - 1]:
                                st.markdown(f"""
                                <a href="{rurl}" target="_blank" style="text-decoration:none;">
                                <div style="background:#242424; border:1px solid #333333; border-radius:12px;
                                            padding:1.1rem 1rem; cursor:pointer; height:100%;">
                                    <div style="font-size:1.5rem; margin-bottom:0.4rem;">{icons[r-1]}</div>
                                    <div style="font-weight:600; color:#eeeeee; font-size:0.92rem; margin-bottom:0.3rem;">{rname}</div>
                                    <div style="font-size:0.76rem; color:#666666;">{rurl.replace("https://","").split("/")[0]}</div>
                                    <div style="margin-top:0.6rem; font-size:0.78rem; color:#888888; font-weight:500;">Open Resource →</div>
                                </div>
                                </a>
                                """, unsafe_allow_html=True)

                # ── Other Suggestions ──
                if len(top_skills) > 1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-header">OTHER GOOD MATCHES</div>', unsafe_allow_html=True)
                    cols = st.columns(min(4, len(top_skills) - 1))
                    for i, (skill, conf) in enumerate(top_skills[1:5]):
                        details = get_skill_details(skill, df)
                        if details is not None:
                            with cols[i % len(cols)]:
                                st.markdown(f"""
                                <div style="background:#242424; border:1px solid #333333;
                                            border-radius:12px; padding:1rem; height:100%;">
                                    <div style="font-weight:600; color:#eeeeee; margin-bottom:0.4rem; font-size:0.93rem;">{skill}</div>
                                    {difficulty_badge(details['difficulty'])}
                                    {cost_badge(details['cost_level'])}
                                    <div style="font-size:0.78rem; color:#888888; margin-top:0.5rem;">⏱ {details['weekly_hours_needed']}h/week</div>
                                    <div style="font-size:0.75rem; color:#666666; margin-top:0.2rem;">{int(conf*100)}% match</div>
                                </div>
                                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 2rem; color:#776699;">
            <div style="font-size:3rem; margin-bottom:1rem; filter: drop-shadow(0 0 16px #7c3aed88);">✨</div>
            <div style="font-size:1.15rem; color:#d4c8f0; margin-bottom:0.5rem; font-weight:700;">Ready when you are!</div>
            <div style="font-size:0.92rem; color:#9988bb;">Set your interests, time, and budget on the left, then click <strong style="color:#a78bfa;">Suggest My Next Skill</strong></div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# USER SUGGESTION SECTION — "I want to learn..."
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:#242424; border:1px solid #333333; border-radius:16px; padding:2rem 2.5rem; margin-bottom:1rem;">
    <div style="font-family:'DM Sans',sans-serif; font-size:1.3rem; font-weight:700; color:#eeeeee; margin-bottom:0.4rem;">
        Have something specific in mind?
    </div>
    <div style="font-size:0.9rem; color:#888888;">
        Tell us what you want to learn and we'll find matching resources and build a step-by-step plan for you.
    </div>
</div>
""", unsafe_allow_html=True)

user_topic = st.text_input(
    "What do you want to learn?",
    placeholder="e.g. I want to learn machine learning, web development, photography...",
    key="user_topic_input"
)
search_btn = st.button("🔍 Find Resources & Plan", key="search_btn")

if search_btn and user_topic.strip():
    # First check if it matches a skill in our database
    topic_lower = user_topic.lower().strip()
    matched = df[df["skill_name"].str.lower().str.contains(topic_lower, na=False) |
                 df["description"].str.lower().str.contains(topic_lower, na=False) |
                 df["category"].str.lower().str.contains(topic_lower, na=False) |
                 df["interest_tags"].str.lower().str.contains(topic_lower, na=False)]

    st.markdown("<br>", unsafe_allow_html=True)

    if not matched.empty:
        skill_row = matched.iloc[0]
        skill_name = skill_row["skill_name"]

        st.markdown(f'<div class="section-header">RESOURCES FOR: {skill_name.upper()}</div>', unsafe_allow_html=True)

        # Resource cards
        res_cols = st.columns(3)
        icons = ["🎓", "📘", "▶️"]
        for r in range(1, 4):
            rname = skill_row.get(f"resource_{r}_name", "")
            rurl = skill_row.get(f"resource_{r}_url", "")
            if rname and rurl and str(rname) != "nan":
                with res_cols[r - 1]:
                    st.markdown(f"""
                    <a href="{rurl}" target="_blank" style="text-decoration:none;">
                    <div style="background:#242424; border:1px solid #333333; border-radius:12px;
                                padding:1.1rem 1rem; cursor:pointer; height:100%;">
                        <div style="font-size:1.5rem; margin-bottom:0.4rem;">{icons[r-1]}</div>
                        <div style="font-weight:600; color:#eeeeee; font-size:0.92rem; margin-bottom:0.3rem;">{rname}</div>
                        <div style="font-size:0.76rem; color:#666666;">{rurl.replace("https://","").split("/")[0]}</div>
                        <div style="margin-top:0.6rem; font-size:0.78rem; color:#888888;">Open Resource →</div>
                    </div>
                    </a>
                    """, unsafe_allow_html=True)

        # Step-by-step learning plan
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">STEP-BY-STEP PLAN FOR {skill_name.upper()}</div>', unsafe_allow_html=True)

        prereq = str(skill_row["prerequisite_skills"])
        diff = skill_row["difficulty"]
        hours = skill_row["weekly_hours_needed"]
        cost = skill_row["cost_level"]

        # Build dynamic plan steps
        plan_steps = []
        if prereq and prereq.lower() != "none":
            plan_steps.append(("01", f"Complete the prerequisite: {prereq}",
                               f"Make sure you have a solid foundation in {prereq} before starting. This will make learning {skill_name} much smoother."))
        plan_steps.append(("02" if prereq.lower() != "none" else "01",
                           "Set up your learning environment",
                           f"Dedicate {hours} hours per week. Find a consistent time slot each day and eliminate distractions. {'This course is free so you can start immediately.' if cost == 'Free' else 'Budget for the course cost and consider it an investment.'}"))

        step_n = len(plan_steps) + 1
        plan_steps.append((f"0{step_n}", f"Start with the basics of {skill_name}",
                           f"Begin with the recommended resources above. Focus on understanding core concepts before moving to practice."))
        step_n += 1
        plan_steps.append((f"0{step_n}", "Build a small project",
                           f"Apply what you've learned by building something small and real. Projects reinforce learning far better than passive study."))
        step_n += 1
        plan_steps.append((f"0{step_n}", "Review, practice, and repeat",
                           f"Go back over difficult concepts. Practice daily. At {hours}h/week, you can expect solid progress within 4–8 weeks."))
        step_n += 1
        plan_steps.append((f"0{step_n}", "Share your work and keep growing",
                           f"Share what you built online (GitHub, LinkedIn, portfolio). This builds credibility and opens up opportunities."))

        for n, title, desc in plan_steps:
            st.markdown(f"""
            <div style="display:flex; gap:1rem; align-items:flex-start; margin-bottom:0.8rem;
                        padding:0.9rem 1.1rem; background:#242424; border:1px solid #333333; border-radius:10px;">
                <div style="background:#3a3a3a; color:#eeeeee; font-weight:700; font-size:0.82rem;
                            border-radius:50%; width:2rem; height:2rem; display:flex; align-items:center;
                            justify-content:center; flex-shrink:0; min-width:2rem;">{n}</div>
                <div>
                    <div style="font-weight:600; color:#eeeeee; margin-bottom:0.2rem; font-size:0.95rem;">{title}</div>
                    <div style="font-size:0.87rem; color:#888888; line-height:1.55;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Topic not in database — show a friendly not-found message
        st.markdown(f"""
        <div style="background:#242424; border:1px solid #333333; border-radius:12px; padding:1.5rem 1.8rem;">
            <div style="font-weight:600; color:#eeeeee; margin-bottom:0.5rem; font-size:1rem;">
                We don't have "{user_topic}" in our database yet
            </div>
            <div style="font-size:0.9rem; color:#888888; line-height:1.6;">
                Try searching for a related topic, or use the AI recommendation above by selecting your interests.
                You can also try terms like: <em>Python, design, marketing, data, photography, finance</em> etc.
            </div>
        </div>
        """, unsafe_allow_html=True)

elif search_btn and not user_topic.strip():
    st.warning("Please type something you want to learn first.")
