import os
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="MaintainIQ | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_USERNAME = "admin"
APP_PASSWORD = "maintainiq"

MODEL_FILES = {
    "preprocessor": "preprocessor.pkl",
    "failure": "best_failure_model.keras",
    "failure_type": "best_failure_type_model.pkl",
    "encoder": "failure_type_encoder.pkl",
    "rul": "best_rul_model.pkl",
    "repair_cost": "best_repair_cost_model.pkl",
}

MODEL_INPUT_FEATURES = [
    "machine_id",
    "machine_type",
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "operating_mode",
    "hours_since_maintenance",
    "ambient_temp",
    "hour",
    "day",
    "month",
    "day_of_week",
]

MACHINE_TYPES = ["CNC", "lathe", "milling", "press", "pump", "compressor"]
OPERATING_MODES = ["idle", "normal", "maintenance", "heavy_load"]

WINNERS = {
    "Failure Prediction": {
        "model": "MLP",
        "metric": "F1 Score",
        "score": 0.780890,
        "artifact": "best_failure_model.keras",
    },
    "Failure Type": {
        "model": "Balanced Extra Trees",
        "metric": "Macro F1",
        "score": 0.8007,
        "artifact": "best_failure_type_model.pkl",
    },
    "Remaining Useful Life": {
        "model": "XGBoost",
        "metric": "MAE",
        "score": 9.2303,
        "artifact": "best_rul_model.pkl",
    },
    "Estimated Repair Cost": {
        "model": "Random Forest",
        "metric": "MAE",
        "score": 286.8328,
        "artifact": "best_repair_cost_model.pkl",
    },
}

FALLBACKS = {
    "failure": pd.DataFrame(
        [["MLP", 0.780890, np.nan, np.nan, "🏆 Selected"]],
        columns=["Model", "F1 Score", "Accuracy", "ROC-AUC", "Status"],
    ),
    "failure_type": pd.DataFrame(
        [
            ["Balanced Extra Trees", 0.8007, 0.9561, "🏆 Selected"],
            ["Balanced Subsample RF", 0.7142, 0.9507, ""],
            ["Current Random Forest", 0.6823, 0.9511, ""],
            ["Balanced Random Forest", 0.6729, 0.9509, ""],
        ],
        columns=["Model", "Macro F1", "Accuracy", "Status"],
    ),
    "rul": pd.DataFrame(
        [["XGBoost", 9.2303, 13.2020, 0.7650, "🏆 Selected"]],
        columns=["Model", "MAE (hours)", "RMSE (hours)", "R²", "Status"],
    ),
    "repair": pd.DataFrame(
        [["Random Forest", 286.8328, 572.4620, 0.5540, "🏆 Selected"]],
        columns=["Model", "MAE (₹)", "RMSE (₹)", "R²", "Status"],
    ),
}

BASE_DIR = Path(__file__).resolve().parent

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html,body,[class*="css"]{
    font-family:'Inter',sans-serif;
}

.stApp{
    background:#f5f7fb !important;
    color:#172033;
}

.block-container{
    max-width:1480px !important;
    padding-top:1.15rem !important;
    padding-bottom:2.5rem !important;
}

[data-testid="stHeader"]{
    height:0 !important;
    background:transparent !important;
}

[data-testid="stDecoration"]{
    display:none !important;
}

[data-testid="stToolbar"]{
    top:.3rem;
}

/* SIDEBAR */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#08172d 0%,#0b2341 100%) !important;
    border-right:1px solid rgba(255,255,255,.08) !important;
    min-width:235px !important;
    max-width:235px !important;
}

[data-testid="stSidebar"] > div:first-child{
    padding-top:.35rem !important;
}

[data-testid="stSidebar"] .stButton{
    margin:0 !important;
    padding:0 !important;
}

[data-testid="stSidebar"] .stButton > button{
    width:100% !important;
    min-height:40px !important;
    height:40px !important;
    margin:2px 0 !important;
    padding:0 13px !important;
    border-radius:10px !important;
    background:transparent !important;
    border:1px solid transparent !important;
    color:#b9c7dc !important;
    text-align:left !important;
    font-size:12px !important;
    font-weight:700 !important;
    box-shadow:none !important;
    transition:.18s ease !important;
}

[data-testid="stSidebar"] .stButton > button:hover{
    background:rgba(56,189,248,.11) !important;
    border-color:rgba(56,189,248,.18) !important;
    color:#fff !important;
    transform:none !important;
}

[data-testid="stSidebar"] .stButton > button:focus{
    box-shadow:none !important;
}

.sidebar-brand{
    text-align:center;
    padding:4px 2px 13px;
}

.sidebar-logo{
    width:52px;
    height:52px;
    margin:auto;
    border-radius:16px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:linear-gradient(135deg,#2563eb,#0891b2);
    color:white;
    font-size:25px;
    box-shadow:0 9px 24px rgba(37,99,235,.25);
}

.sidebar-title{
    color:#fff;
    font-size:21px;
    font-weight:850;
    margin-top:7px;
}

.sidebar-subtitle{
    color:#8fa5c6;
    font-size:9px;
    line-height:1.35;
    margin-top:2px;
}

.admin-card{
    padding:12px;
    border-radius:15px;
    background:rgba(255,255,255,.065);
    border:1px solid rgba(255,255,255,.09);
    margin:2px 0 13px;
}

.admin-row{
    display:flex;
    align-items:center;
    gap:9px;
}

.admin-avatar{
    width:39px;
    height:39px;
    border-radius:12px;
    background:linear-gradient(135deg,#dbeafe,#ede9fe);
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:19px;
}

.admin-name{
    color:#fff;
    font-size:12px;
    font-weight:800;
}

.admin-role{
    color:#8fa5c6;
    font-size:9px;
    margin-top:2px;
}

.online-badge{
    display:inline-block;
    margin-top:9px;
    padding:4px 8px;
    border-radius:20px;
    background:rgba(34,197,94,.12);
    border:1px solid rgba(34,197,94,.25);
    color:#4ade80;
    font-size:8px;
    font-weight:800;
}

.sidebar-label{
    color:#7183a8;
    font-size:8px;
    font-weight:800;
    letter-spacing:1.1px;
    margin:0 4px 6px;
}

.sidebar-spacer{
    height:5px;
}

/* LOGIN */
.login-wrapper{
    max-width:470px;
    margin:35px auto 0;
}

.login-brand{
    background:linear-gradient(135deg,#0b1f38,#173c5c);
    border-radius:24px;
    padding:32px 30px;
    text-align:center;
    border:1px solid rgba(255,255,255,.1);
    box-shadow:0 18px 45px rgba(8,23,45,.14);
}

.login-logo{
    width:62px;
    height:62px;
    margin:0 auto 14px;
    border-radius:18px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:linear-gradient(135deg,#2563eb,#0891b2);
    color:#fff;
    font-size:29px;
    box-shadow:0 10px 25px rgba(37,99,235,.28);
}

.login-title{
    color:#fff;
    font-size:28px;
    font-weight:850;
    letter-spacing:-.7px;
}

.login-subtitle{
    color:#b8c9df;
    font-size:12px;
    margin-top:5px;
}

.login-form{
    margin-top:20px;
}

.login-heading{
    color:#14213d;
    font-size:30px;
    font-weight:850;
    margin-bottom:5px;
}

.login-caption{
    color:#718096;
    font-size:13px;
    margin-bottom:17px;
}

.login-demo{
    background:#e8f1ff;
    color:#1457a6;
    border:1px solid #d5e5fb;
    border-radius:12px;
    padding:12px 15px;
    font-size:12px;
    margin-top:11px;
}

/* GENERAL */
.page-title{
    color:#14213d !important;
    font-size:32px !important;
    font-weight:850 !important;
    letter-spacing:-1px !important;
    margin-bottom:1px !important;
}

.page-subtitle{
    color:#718096 !important;
    font-size:14px !important;
    margin-bottom:20px !important;
}

.hero{
    min-height:285px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    padding:43px 48px;
    margin-bottom:22px;
    border-radius:25px;
    overflow:hidden;
    position:relative;
    color:#fff;
    background:linear-gradient(110deg,#07172c 0%,#0d3150 60%,#174d67 100%);
    box-shadow:0 17px 45px rgba(8,23,45,.15);
    border:1px solid rgba(255,255,255,.08);
}

.hero-tag{
    background:rgba(37,99,235,.2);
    border:1px solid rgba(125,211,252,.25);
    color:#dff7ff;
    width:max-content;
    padding:7px 11px;
    border-radius:999px;
    font-size:9px;
    font-weight:800;
    letter-spacing:.45px;
}

.hero h1{
    max-width:820px;
    color:#fff;
    font-size:clamp(34px,4vw,52px);
    line-height:1.03;
    letter-spacing:-1.7px;
    margin:7px 0 12px;
}

.hero p{
    max-width:720px;
    color:rgba(255,255,255,.82);
    font-size:14px;
    line-height:1.7;
}

div[data-testid="stMetric"]{
    background:#fff !important;
    border:1px solid #e7ebf3 !important;
    border-radius:16px !important;
    padding:15px 17px !important;
    box-shadow:0 7px 23px rgba(16,24,40,.05) !important;
}

div[data-testid="stMetric"] label{
    color:#78859b !important;
    font-size:10px !important;
    font-weight:750 !important;
}

div[data-testid="stMetricValue"]{
    color:#14213d !important;
    font-weight:850 !important;
}

.pred{
    background:#fff !important;
    border:1px solid #e6eaf2 !important;
    border-radius:18px !important;
    min-height:155px !important;
    padding:19px !important;
    box-shadow:0 9px 27px rgba(16,24,40,.05) !important;
    transition:.18s ease !important;
}

.pred:hover{
    transform:translateY(-2px) !important;
    box-shadow:0 13px 33px rgba(16,24,40,.08) !important;
}

.pred-icon{
    font-size:26px !important;
}

.pred-label{
    color:#7b879b !important;
    font-size:9px !important;
    font-weight:800 !important;
    letter-spacing:.65px !important;
    text-transform:uppercase;
    margin-top:6px;
}

.pred-value{
    color:#13213b !important;
    font-size:24px !important;
    font-weight:850 !important;
    margin-top:4px;
}

.badge{
    display:inline-block;
    margin-top:6px;
    padding:5px 9px;
    border-radius:999px;
    font-size:9px;
    font-weight:800;
}

.green{
    background:#d1fae5;
    color:#047857;
}

.orange{
    background:#fef3c7;
    color:#b45309;
}

.red{
    background:#fee2e2;
    color:#b91c1c;
}

.blue{
    background:#dbeafe;
    color:#1d4ed8;
}

.winner{
    padding:14px;
    border-radius:15px;
    margin-bottom:8px;
    border:1px solid #e2e8f2;
    background:linear-gradient(135deg,#fff,#f8fbff);
    box-shadow:0 6px 18px rgba(37,99,235,.045);
}

.small{
    color:#64748b;
    font-size:10px;
}

.section-card{
    background:#fff;
    border:1px solid #e6eaf2;
    border-radius:18px;
    padding:22px;
    box-shadow:0 8px 25px rgba(16,24,40,.045);
}

.section-head{
    color:#14213d;
    font-size:19px;
    font-weight:800;
}

.section-note{
    color:#71809a;
    font-size:11px;
    margin:4px 0 15px;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stDateInput input,
.stTimeInput input{
    background:#fff !important;
    border:1px solid #dbe2ec !important;
    border-radius:10px !important;
    min-height:41px !important;
}

.stTextInput input:focus,
.stNumberInput input:focus{
    border-color:#60a5fa !important;
    box-shadow:0 0 0 3px rgba(37,99,235,.09) !important;
}

.stButton > button,
.stFormSubmitButton > button{
    border-radius:11px !important;
    min-height:43px !important;
    font-weight:800 !important;
    border:1px solid #dbe3ef !important;
    box-shadow:0 5px 14px rgba(16,24,40,.04) !important;
}

.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"]{
    background:linear-gradient(135deg,#2563eb,#0891b2) !important;
    color:#fff !important;
    border:0 !important;
    box-shadow:0 9px 22px rgba(37,99,235,.2) !important;
}

.stAlert{
    border-radius:13px !important;
}

[data-testid="stDataFrame"]{
    border:1px solid #e4e9f1 !important;
    border-radius:13px !important;
    overflow:hidden !important;
    box-shadow:0 7px 22px rgba(16,24,40,.04) !important;
}

.footer{
    color:#94a0b4;
    font-size:11px;
    margin-top:28px;
    padding-top:15px;
    border-top:1px solid #e5e9f0;
}

@media(max-width:900px){
    [data-testid="stSidebar"]{
        min-width:220px !important;
        max-width:220px !important;
    }

    .block-container{
        padding-left:1rem !important;
        padding-right:1rem !important;
    }

    .hero{
        padding:32px 26px;
        min-height:250px;
    }

    .hero h1{
        font-size:35px;
    }

    .login-wrapper{
        margin-top:20px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

def find_file(filename):
    folders = [
        BASE_DIR / "maintainiq_models",
        BASE_DIR / "models",
        BASE_DIR,
        Path.cwd() / "maintainiq_models",
        Path.cwd() / "models",
        Path.cwd(),
    ]
    seen = set()
    for folder in folders:
        folder = Path(folder).resolve()
        if folder in seen:
            continue
        seen.add(folder)
        path = folder / filename
        if path.exists():
            return path
    return None

def model_paths():
    return {k: find_file(v) for k, v in MODEL_FILES.items()}

@st.cache_resource(show_spinner=False)
def load_models():
    paths = model_paths()
    missing = [MODEL_FILES[k] for k, v in paths.items() if v is None]
    if missing:
        raise FileNotFoundError(
            "Missing deployment files: " + ", ".join(missing)
        )
    return {
        "preprocessor": joblib.load(paths["preprocessor"]),
        "failure": load_model(paths["failure"], compile=False),
        "failure_type": joblib.load(paths["failure_type"]),
        "encoder": joblib.load(paths["encoder"]),
        "rul": joblib.load(paths["rul"]),
        "repair_cost": joblib.load(paths["repair_cost"]),
        "paths": paths,
    }

for key, default in {
    "authenticated": False,
    "page": "Home",
    "history": [],
    "last_prediction": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def login_page():
    st.markdown(
        """
        <div class="login-wrapper">
            <div class="login-brand">
                <div class="login-logo">⚙️</div>
                <div class="login-title">MaintainIQ</div>
                <div class="login-subtitle">
                    Intelligent Predictive Maintenance
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-wrapper login-form">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-heading">🔐 Welcome back</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-caption">'
        'Sign in to access your predictive maintenance control center.'
        '</div>',
        unsafe_allow_html=True,
    )

    username = st.text_input(
        "Username",
        placeholder="Enter username",
        key="login_username",
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter password",
        key="login_password",
    )

    if st.button(
        "🚀  Sign In to MaintainIQ",
        type="primary",
        use_container_width=True,
        key="login_button",
    ):
        if (
            username.strip().lower() == APP_USERNAME
            and password == APP_PASSWORD
        ):
            st.session_state.authenticated = True
            st.session_state.page = "Home"
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.markdown(
        '<div class="login-demo">'
        'Demo access&nbsp;&nbsp;•&nbsp;&nbsp;admin / maintainiq'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.authenticated:
    login_page()
    st.stop()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">⚙️</div>
            <div class="sidebar-title">MaintainIQ</div>
            <div class="sidebar-subtitle">
                Predictive Maintenance Intelligence
            </div>
        </div>

        <div class="admin-card">
            <div class="admin-row">
                <div class="admin-avatar">👨‍💼</div>
                <div>
                    <div class="admin-name">Hello, Admin</div>
                    <div class="admin-role">Administrator</div>
                </div>
            </div>
            <div class="online-badge">● SYSTEM ONLINE</div>
        </div>

        <div class="sidebar-label">CONTROL CENTER</div>
        """,
        unsafe_allow_html=True,
    )

    nav = [
        ("🏠", "Home"),
        ("📊", "Dashboard"),
        ("🔮", "Predictions"),
        ("📜", "History"),
        ("📈", "Analytics"),
        ("🤖", "Model Performance"),
        ("ℹ️", "About"),
    ]

    for icon, label in nav:
        if st.button(
            f"{icon}   {label}",
            key=f"nav_{label}",
            use_container_width=True,
        ):
            st.session_state.page = label
            st.rerun()

    st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)

    if st.button(
        "🚪   Logout",
        key="logout",
        use_container_width=True,
    ):
        st.session_state.authenticated = False
        st.session_state.page = "Home"
        st.session_state.history = []
        st.session_state.last_prediction = None
        st.rerun()

try:
    models = load_models()
except Exception as exc:
    st.error("MaintainIQ deployment models could not be loaded.")
    st.exception(exc)
    st.info(
        "Keep app.py together with a maintainiq_models folder containing "
        "the six required deployment artifacts."
    )
    st.stop()

def expected_features():
    pre = models["preprocessor"]

    names = getattr(pre, "feature_names_in_", None)
    if names is not None:
        return [str(x) for x in names]

    found = []
    try:
        for _, _, columns in pre.transformers_:
            if isinstance(
                columns,
                (list, tuple, np.ndarray, pd.Index),
            ):
                found.extend(str(x) for x in columns)
    except Exception:
        pass

    return (
        list(dict.fromkeys(found))
        if found
        else MODEL_INPUT_FEATURES.copy()
    )

def prepare_input(
    machine_id,
    machine_type,
    vibration_rms,
    temperature_motor,
    current_phase_avg,
    pressure_level,
    rpm,
    operating_mode,
    hours_since_maintenance,
    ambient_temp,
    prediction_dt,
):
    values = {
        "machine_id": machine_id,
        "machine_type": machine_type,
        "vibration_rms": vibration_rms,
        "temperature_motor": temperature_motor,
        "current_phase_avg": current_phase_avg,
        "pressure_level": pressure_level,
        "rpm": rpm,
        "operating_mode": operating_mode,
        "hours_since_maintenance": hours_since_maintenance,
        "ambient_temp": ambient_temp,
        "hour": prediction_dt.hour,
        "day": prediction_dt.day,
        "month": prediction_dt.month,
        "day_of_week": prediction_dt.weekday(),
    }

    expected = expected_features()

    missing = [x for x in expected if x not in values]

    if missing:
        raise ValueError(
            "The saved preprocessor expects features not supplied by the UI: "
            + str(missing)
        )

    return pd.DataFrame(
        [[values[x] for x in expected]],
        columns=expected,
    )

def prediction(input_df):
    processed = models["preprocessor"].transform(input_df)

    raw = np.asarray(
        models["failure"].predict(processed, verbose=0)
    )

    if raw.ndim == 2 and raw.shape[1] == 2:
        probability = float(raw[0, 1])
    else:
        probability = float(raw.reshape(-1)[0])

    probability = float(np.clip(probability, 0, 1))

    failure_decision = (
        "Failure Risk"
        if probability >= 0.50
        else "No Failure"
    )

    raw_type = np.asarray(
        models["failure_type"].predict(processed)
    ).reshape(-1)[0]

    if isinstance(raw_type, str):
        failure_type = raw_type
    else:
        failure_type = str(
            models["encoder"].inverse_transform(
                [int(raw_type)]
            )[0]
        )

    failure_type = failure_type.strip().lower()

    rul = float(
        np.asarray(
            models["rul"].predict(processed)
        ).reshape(-1)[0]
    )
    rul = max(0.0, rul)

    cost = float(
        np.asarray(
            models["repair_cost"].predict(processed)
        ).reshape(-1)[0]
    )
    cost = max(0.0, cost)

    if probability >= 0.50:
        risk, risk_class = "High", "red"
    elif probability >= 0.20:
        risk, risk_class = "Moderate", "orange"
    else:
        risk, risk_class = "Low", "green"

    return {
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "machine_id": int(input_df.iloc[0]["machine_id"]),
        "machine_type": str(
            input_df.iloc[0]["machine_type"]
        ),
        "failure_probability": probability,
        "failure_prediction": failure_decision,
        "risk_level": risk,
        "risk_class": risk_class,
        "failure_type": failure_type,
        "rul_hours": rul,
        "repair_cost": cost,
    }

def save_result(result):
    st.session_state.last_prediction = result
    st.session_state.history.insert(0, result)
    st.session_state.history = st.session_state.history[:100]

def cards(result):
    cols = st.columns(4)

    data = [
        (
            "⚠️",
            "Failure Probability",
            f'{result["failure_probability"] * 100:.2f}%',
            f'{result["risk_level"]} Risk',
            result["risk_class"],
        ),
        (
            "🔧",
            "Failure Type",
            result["failure_type"]
            .replace("_", " ")
            .title(),
            "Model classified",
            "blue",
        ),
        (
            "⏱️",
            "Remaining Useful Life",
            f'{result["rul_hours"]:.2f} h',
            "Estimated remaining life",
            "blue",
        ),
        (
            "💰",
            "Estimated Repair Cost",
            f'₹{result["repair_cost"]:,.2f}',
            "Model estimate",
            "green",
        ),
    ]

    for col, item in zip(cols, data):
        icon, label, value, note, color = item
        with col:
            st.markdown(
                f"""
                <div class="pred">
                    <div class="pred-icon">{icon}</div>
                    <div class="pred-label">{label}</div>
                    <div class="pred-value">{value}</div>
                    <span class="badge {color}">{note}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

def header(title, subtitle):
    st.markdown(
        f'<div class="page-title">{title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )

def home():
    st.markdown(
        """
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin:0 0 11px 2px">
            <span style="background:#ecfdf3;color:#087443;border:1px solid #ccefdc;padding:6px 10px;border-radius:999px;font-size:9px;font-weight:800">
                ● SYSTEM ONLINE
            </span>
            <span style="background:#eff6ff;color:#1d4ed8;border:1px solid #dbeafe;padding:6px 10px;border-radius:999px;font-size:9px;font-weight:800">
                ⚙ 6 MODEL ARTIFACTS READY
            </span>
            <span style="background:#f5f3ff;color:#6d28d9;border:1px solid #e9d5ff;padding:6px 10px;border-radius:999px;font-size:9px;font-weight:800">
                AI MONITORING ACTIVE
            </span>
        </div>

        <div class="hero">
            <div class="hero-tag">
                ⚙️ SMART FACTORY • PREDICTIVE MAINTENANCE
            </div>
            <h1>Predict problems before machines stop.</h1>
            <p>
                AI-powered machine intelligence that predicts failure risk,
                failure type, remaining useful life and estimated repair cost
                from current machine conditions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    h = st.session_state.history
    total = len(h)
    avg_risk = (
        np.mean([x["failure_probability"] for x in h]) * 100
        if h
        else 0
    )
    high = sum(
        x["failure_probability"] >= 0.50
        for x in h
    )
    total_cost = sum(
        x["repair_cost"]
        for x in h
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predictions", total)
    c2.metric("High Risk", high)
    c3.metric("Average Risk", f"{avg_risk:.2f}%")
    c4.metric("Estimated Cost", f"₹{total_cost:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    a, b = st.columns([1.3, 1])

    with a:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-head">🚀 What MaintainIQ Does</div>
                <div style="color:#64748b;font-size:13px;line-height:1.65;margin-top:8px">
                    MaintainIQ converts machine-health and operating information
                    into practical predictive-maintenance insights.
                </div>
                <div style="margin-top:13px;color:#344054;font-size:12px;line-height:2">
                    ⚠️ <b>Failure prediction</b> within 24 hours<br>
                    🔧 <b>Failure type</b> classification<br>
                    ⏱️ <b>RUL</b> estimation in hours<br>
                    💰 <b>Repair cost</b> estimation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.success("All six deployment artifacts are loaded.")

    with b:
        st.markdown(
            '<div class="section-head" style="margin-bottom:9px">'
            '🏆 Best Models'
            '</div>',
            unsafe_allow_html=True,
        )

        for task, info in WINNERS.items():
            st.markdown(
                f"""
                <div class="winner">
                    <div class="small">{task}</div>
                    <b>🏆 {info["model"]}</b><br>
                    <span class="small">
                        {info["metric"]}: {info["score"]}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    x, y = st.columns(2)

    with x:
        if st.button(
            "🔮 Make New Prediction",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.page = "Predictions"
            st.rerun()

    with y:
        if st.button(
            "📊 Open Dashboard",
            use_container_width=True,
        ):
            st.session_state.page = "Dashboard"
            st.rerun()

def dashboard():
    header(
        "📊 Operations Dashboard",
        "A colorful command center for machine risk, RUL and maintenance cost.",
    )

    if not st.session_state.history:
        st.markdown(
            """
            <div class="section-card" style="text-align:center;padding:45px">
                <div style="font-size:42px">📡</div>
                <div class="section-head">No prediction data yet</div>
                <div class="section-note">
                    Generate a prediction to populate your dashboard.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(st.session_state.history)

    high = int(
        (df.failure_probability >= 0.50).sum()
    )
    moderate = int(
        (
            (df.failure_probability >= 0.20)
            & (df.failure_probability < 0.50)
        ).sum()
    )
    low = len(df) - high - moderate

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔮 Total Predictions", len(df))
    c2.metric("🔴 High Risk", high)
    c3.metric("🟠 Moderate Risk", moderate)
    c4.metric("🟢 Low Risk", low)

    st.markdown(
        "<div style='height:12px'></div>",
        unsafe_allow_html=True,
    )

    a, b = st.columns(2)

    with a:
        risk = pd.DataFrame(
            {
                "Risk": ["Low", "Moderate", "High"],
                "Machines": [low, moderate, high],
            }
        )

        fig = px.pie(
            risk,
            names="Risk",
            values="Machines",
            hole=0.62,
            color="Risk",
            color_discrete_map={
                "Low": "#10b981",
                "Moderate": "#f59e0b",
                "High": "#ef4444",
            },
            title="Risk Distribution",
        )

        fig.add_annotation(
            text=f"<b>{len(df)}</b><br>Predictions",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=16,
                color="#14213d",
            ),
        )

        fig.update_layout(
            height=410,
            margin=dict(
                l=15,
                r=15,
                t=55,
                b=15,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with b:
        top = df.sort_values(
            "repair_cost",
            ascending=False,
        ).head(10)

        fig = px.bar(
            top,
            x="machine_id",
            y="repair_cost",
            color="repair_cost",
            color_continuous_scale=[
                "#2563eb",
                "#7c3aed",
                "#ef4444",
            ],
            title="Top Estimated Repair Costs",
            labels={
                "machine_id": "Machine ID",
                "repair_cost": "Repair Cost (₹)",
            },
        )

        fig.update_layout(
            height=410,
            coloraxis_showscale=False,
            margin=dict(
                l=15,
                r=15,
                t=55,
                b=15,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    a, b = st.columns(2)

    with a:
        counts = (
            df.machine_type
            .value_counts()
            .reset_index()
        )
        counts.columns = [
            "Machine Type",
            "Count",
        ]

        fig = px.bar(
            counts,
            x="Machine Type",
            y="Count",
            color="Machine Type",
            title="Prediction Volume by Machine Type",
        )

        fig.update_layout(
            showlegend=False,
            height=370,
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with b:
        fig = px.scatter(
            df,
            x="rul_hours",
            y="repair_cost",
            color="risk_level",
            size=np.maximum(
                df["failure_probability"] * 100,
                5,
            ),
            hover_data=[
                "machine_id",
                "machine_type",
                "failure_type",
            ],
            color_discrete_map={
                "Low": "#10b981",
                "Moderate": "#f59e0b",
                "High": "#ef4444",
            },
            title="RUL vs Estimated Repair Cost",
            labels={
                "rul_hours": "RUL (hours)",
                "repair_cost": "Repair Cost (₹)",
                "risk_level": "Risk",
            },
        )

        fig.update_layout(
            height=370,
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown("### 📈 Risk Trend")

    timeline = df.iloc[::-1].copy()

    fig = px.line(
        timeline,
        x="timestamp",
        y="failure_probability",
        markers=True,
        title="Failure Probability Across Prediction Events",
        labels={
            "timestamp": "Time",
            "failure_probability": "Failure Probability",
        },
    )

    fig.update_yaxes(
        tickformat=".0%",
        range=[0, 1],
    )

    fig.update_traces(line_width=3)

    fig.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown("### 🧾 Recent Prediction Activity")

    view = df.copy()

    view["Failure Probability"] = (
        view.failure_probability * 100
    ).round(2).astype(str) + "%"

    view["Failure Type"] = (
        view.failure_type
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    view = view[
        [
            "timestamp",
            "machine_id",
            "machine_type",
            "Failure Probability",
            "failure_prediction",
            "Failure Type",
            "risk_level",
            "rul_hours",
            "repair_cost",
        ]
    ].rename(
        columns={
            "timestamp": "Time",
            "machine_id": "Machine ID",
            "machine_type": "Machine Type",
            "failure_prediction": "Prediction",
            "risk_level": "Risk",
            "rul_hours": "RUL (Hours)",
            "repair_cost": "Repair Cost (₹)",
        }
    )

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
    )

def predictions():
    header(
        "🔮 Predictions",
        "Enter current machine conditions and generate four ML predictions.",
    )

    st.info(
        "Pipeline: machine inputs → saved preprocessor → selected deployment "
        "models → risk, type, RUL and repair-cost predictions."
    )

    with st.form("prediction_form"):
        st.markdown("### ⚙️ Machine Information")

        a, b, c = st.columns(3)

        with a:
            machine_id = st.number_input(
                "Machine ID",
                min_value=1,
                value=941,
                step=1,
            )

            machine_type = st.selectbox(
                "Machine Type",
                MACHINE_TYPES,
            )

        with b:
            operating_mode = st.selectbox(
                "Operating Mode",
                OPERATING_MODES,
            )

            hours_since = st.number_input(
                "Hours Since Maintenance",
                min_value=0.0,
                value=147.12,
                step=1.0,
            )

        with c:
            pdate = st.date_input(
                "Prediction Date",
                datetime.now().date(),
            )

            ptime = st.time_input(
                "Prediction Time",
                datetime.now()
                .time()
                .replace(
                    second=0,
                    microsecond=0,
                ),
            )

        st.markdown("### 📡 Sensor Measurements")

        a, b, c = st.columns(3)

        with a:
            vibration = st.number_input(
                "Vibration RMS",
                min_value=0.0,
                value=0.83,
                step=0.01,
            )

            temperature = st.number_input(
                "Motor Temperature",
                value=47.85,
                step=0.1,
            )

        with b:
            current = st.number_input(
                "Average Phase Current",
                min_value=0.0,
                value=4.29,
                step=0.01,
            )

            pressure = st.number_input(
                "Pressure Level",
                min_value=0.0,
                value=24.90,
                step=0.1,
            )

        with c:
            rpm = st.number_input(
                "RPM",
                min_value=0.0,
                value=895.4,
                step=0.1,
            )

            ambient = st.number_input(
                "Ambient Temperature",
                value=10.5,
                step=0.1,
            )

        submitted = st.form_submit_button(
            "🔮 Generate Predictions",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        dt = datetime.combine(
            pdate,
            ptime,
        )

        try:
            X = prepare_input(
                machine_id,
                machine_type,
                vibration,
                temperature,
                current,
                pressure,
                rpm,
                operating_mode,
                hours_since,
                ambient,
                dt,
            )

            with st.spinner(
                "Running MaintainIQ ML pipeline..."
            ):
                result = prediction(X)

            save_result(result)

            st.success(
                "Prediction completed successfully."
            )

            cards(result)

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result["failure_probability"] * 100,
                    number={
                        "suffix": "%",
                        "font": {
                            "size": 30,
                            "color": "#14213d",
                        },
                    },
                    title={
                        "text": "24-hour failure probability",
                        "font": {"size": 14},
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "ticksuffix": "%",
                        },
                        "bar": {
                            "color": "#4f46e5"
                        },
                        "bgcolor": "#eef2ff",
                        "borderwidth": 0,
                        "steps": [
                            {
                                "range": [0, 20],
                                "color": "#dcfce7",
                            },
                            {
                                "range": [20, 50],
                                "color": "#fef3c7",
                            },
                            {
                                "range": [50, 100],
                                "color": "#fee2e2",
                            },
                        ],
                        "threshold": {
                            "line": {
                                "color": "#ef4444",
                                "width": 4,
                            },
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                )
            )

            gauge.update_layout(
                height=310,
                margin=dict(
                    l=20,
                    r=20,
                    t=55,
                    b=10,
                ),
                paper_bgcolor="rgba(0,0,0,0)",
            )

            g1, g2 = st.columns([1, 1.2])

            with g1:
                st.plotly_chart(
                    gauge,
                    use_container_width=True,
                )

            with g2:
                output_df = pd.DataFrame(
                    {
                        "Output": [
                            "RUL (hours)",
                            "Repair Cost (₹)",
                        ],
                        "Value": [
                            result["rul_hours"],
                            result["repair_cost"],
                        ],
                    }
                )

                out_fig = px.bar(
                    output_df,
                    x="Value",
                    y="Output",
                    orientation="h",
                    color="Output",
                    title="Prediction Output Overview",
                )

                out_fig.update_layout(
                    height=310,
                    paper_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(
                    out_fig,
                    use_container_width=True,
                )

            if result["failure_prediction"] == "Failure Risk":
                st.warning(
                    "Maintenance attention is recommended: predicted "
                    "24-hour failure probability is at least 50%."
                )
            else:
                st.success(
                    "No failure is predicted within 24 hours at the "
                    "current model threshold."
                )

            st.markdown(
                "### 💡 Maintenance Insight"
            )

            if result["risk_level"] == "High":
                text = (
                    f"Prioritize inspection for the predicted "
                    f"{result['failure_type'].replace('_', ' ')} condition."
                )
            elif result["risk_level"] == "Moderate":
                text = (
                    "Continue close monitoring and plan preventive maintenance."
                )
            else:
                text = (
                    "Machine is currently in the low-risk range."
                )

            st.info(
                text
                + f" Estimated RUL: {result['rul_hours']:.2f} hours."
            )

            with st.expander(
                "🔍 View exact model input"
            ):
                st.dataframe(
                    X,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as exc:
            st.error("Prediction failed.")
            st.exception(exc)

    elif st.session_state.last_prediction:
        st.markdown("### Latest Prediction")
        cards(
            st.session_state.last_prediction
        )

def history():
    header(
        "📜 Prediction History",
        "Predictions generated during the current application session.",
    )

    if not st.session_state.history:
        st.info(
            "No history yet. Generate a prediction first."
        )
        return

    df = pd.DataFrame(
        st.session_state.history
    )

    a, b = st.columns(2)

    with a:
        if st.button(
            "🗑️ Clear History",
            use_container_width=True,
        ):
            st.session_state.history = []
            st.session_state.last_prediction = None
            st.rerun()

    with b:
        st.download_button(
            "⬇️ Export Prediction History",
            df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="maintainiq_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    view = df.copy()

    view["failure_probability"] = (
        view.failure_probability * 100
    ).round(2).astype(str) + "%"

    view["failure_type"] = (
        view.failure_type
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    view = view.rename(
        columns={
            "timestamp": "Time",
            "machine_id": "Machine ID",
            "machine_type": "Machine Type",
            "failure_probability": "Failure Probability",
            "failure_prediction": "Prediction",
            "failure_type": "Failure Type",
            "risk_level": "Risk",
            "rul_hours": "RUL (Hours)",
            "repair_cost": "Repair Cost (₹)",
        }
    )

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
    )

def analytics():
    header(
        "📈 Visual Analytics",
        "Explore risk, failure types, RUL and cost patterns across predictions.",
    )

    if not st.session_state.history:
        st.markdown(
            """
            <div class="section-card" style="text-align:center;padding:45px">
                <div style="font-size:42px">📊</div>
                <div class="section-head">
                    Analytics will appear here
                </div>
                <div class="section-note">
                    Generate several predictions for richer insights.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(
        st.session_state.history
    )

    avg_prob = (
        df.failure_probability.mean() * 100
    )
    avg_rul = df.rul_hours.mean()
    avg_cost = df.repair_cost.mean()
    max_cost = df.repair_cost.max()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🎯 Average Risk",
        f"{avg_prob:.2f}%",
    )

    c2.metric(
        "⏱️ Average RUL",
        f"{avg_rul:.2f} h",
    )

    c3.metric(
        "💵 Average Cost",
        f"₹{avg_cost:,.0f}",
    )

    c4.metric(
        "💎 Highest Cost",
        f"₹{max_cost:,.0f}",
    )

    a, b = st.columns(2)

    with a:
        fig = px.histogram(
            df,
            x="failure_probability",
            nbins=10,
            color_discrete_sequence=["#6366f1"],
            title="Failure Probability Distribution",
        )

        fig.update_xaxes(
            tickformat=".0%"
        )

        fig.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with b:
        fig = px.scatter(
            df,
            x="rul_hours",
            y="repair_cost",
            color="risk_level",
            size=np.maximum(
                df["failure_probability"] * 100,
                5,
            ),
            hover_data=[
                "machine_id",
                "machine_type",
                "failure_type",
            ],
            color_discrete_map={
                "Low": "#10b981",
                "Moderate": "#f59e0b",
                "High": "#ef4444",
            },
            title="RUL vs Estimated Repair Cost",
            labels={
                "rul_hours": "RUL (hours)",
                "repair_cost": "Repair Cost (₹)",
                "risk_level": "Risk",
            },
        )

        fig.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    a, b = st.columns(2)

    with a:
        counts = (
            df.failure_type
            .value_counts()
            .reset_index()
        )

        counts.columns = [
            "Failure Type",
            "Count",
        ]

        counts["Failure Type"] = (
            counts["Failure Type"]
            .str.replace("_", " ", regex=False)
            .str.title()
        )

        fig = px.bar(
            counts,
            x="Failure Type",
            y="Count",
            color="Failure Type",
            title="Predicted Failure Types",
        )

        fig.update_layout(
            showlegend=False,
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with b:
        machine_counts = (
            df.machine_type
            .value_counts()
            .reset_index()
        )

        machine_counts.columns = [
            "Machine Type",
            "Count",
        ]

        fig = px.pie(
            machine_counts,
            names="Machine Type",
            values="Count",
            hole=0.48,
            title="Prediction Share by Machine Type",
        )

        fig.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        "### 📉 Prediction Timeline"
    )

    timeline = df.iloc[::-1].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timeline["timestamp"],
            y=timeline["failure_probability"] * 100,
            mode="lines+markers",
            name="Failure Probability (%)",
            line=dict(width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=timeline["timestamp"],
            y=timeline["rul_hours"],
            mode="lines+markers",
            name="RUL (hours)",
            yaxis="y2",
            line=dict(
                width=3,
                dash="dot",
            ),
        )
    )

    fig.update_layout(
        height=430,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
        title="Risk and RUL Across Prediction Events",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False,
            title="Prediction Time",
        ),
        yaxis=dict(
            title="Failure Probability (%)",
            gridcolor="rgba(148,163,184,.16)",
        ),
        yaxis2=dict(
            title="RUL (hours)",
            overlaying="y",
            side="right",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "These charts describe current-session predictions, not training/test "
        "evaluation. Official metrics are shown in Model Performance."
    )

def comparison(
    filename,
    fallback,
    selected,
):
    path = find_file(filename)

    if path:
        try:
            df = pd.read_csv(path)

            if not df.empty:
                return df

        except Exception:
            pass

    return fallback.copy()

def performance():
    header(
        "🤖 Model Performance",
        "Candidate comparison, evaluation metrics and final deployment winners.",
    )

    st.success(
        "Each ML task was evaluated independently. The best task-specific "
        "model was selected for deployment."
    )

    st.markdown(
        "### 🏆Best Models"
    )

    cols = st.columns(4)

    for col, (task, info) in zip(
        cols,
        WINNERS.items(),
    ):
        with col:
            st.markdown(
                f"""
                <div class="winner">
                    <div class="small">{task}</div>
                    <b>🏆 {info["model"]}</b><br>
                    <span class="small">
                        {info["metric"]}: {info["score"]}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        "### 📊 Deployment Quality Snapshot"
    )

    quality = pd.DataFrame(
        {
            "Task": [
                "Failure Prediction",
                "Failure Type",
                "RUL",
                "Repair Cost",
            ],
            "Score": [
                0.780890,
                0.8007,
                0.7650,
                0.5540,
            ],
            "Model": [
                "MLP",
                "Balanced Extra Trees",
                "XGBoost",
                "Random Forest",
            ],
        }
    )

    fig = px.bar(
        quality,
        x="Task",
        y="Score",
        color="Model",
        text="Score",
        title="Final Model Quality Snapshot",
    )

    fig.update_yaxes(
        range=[0, 1]
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    fig.update_layout(
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown(
        "### 1️⃣ Failure Prediction"
    )

    st.dataframe(
        comparison(
            "failure_model_comparison.csv",
            FALLBACKS["failure"],
            "MLP",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Primary metric: F1 Score (higher is better). Deployment: MLP."
    )

    st.markdown(
        "### 2️⃣ Failure Type Classification"
    )

    st.dataframe(
        comparison(
            "failure_type_model_comparison.csv",
            FALLBACKS["failure_type"],
            "Balanced Extra Trees",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Primary metric: Macro F1 (higher is better). "
        "Deployment: Balanced Extra Trees."
    )

    st.markdown(
        "### 3️⃣ Remaining Useful Life"
    )

    st.dataframe(
        comparison(
            "rul_model_comparison.csv",
            FALLBACKS["rul"],
            "XGBoost",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Primary metric: MAE (lower is better). Deployment: XGBoost."
    )

    st.markdown(
        "### 4️⃣ Estimated Repair Cost"
    )

    st.dataframe(
        comparison(
            "repair_cost_model_comparison.csv",
            FALLBACKS["repair"],
            "Random Forest",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Primary metric: MAE (lower is better). "
        "Deployment: Random Forest."
    )

    st.markdown(
        "### 📊 Key Evaluation Metrics"
    )

    metrics = pd.DataFrame(
        [
            [
                "Failure Prediction",
                "MLP",
                "F1 Score",
                0.780890,
            ],
            [
                "Failure Type",
                "Balanced Extra Trees",
                "Macro F1",
                0.8007,
            ],
            [
                "Failure Type",
                "Balanced Extra Trees",
                "Accuracy",
                0.9561,
            ],
            [
                "RUL",
                "XGBoost",
                "MAE (hours)",
                9.2303,
            ],
            [
                "RUL",
                "XGBoost",
                "RMSE (hours)",
                13.2020,
            ],
            [
                "RUL",
                "XGBoost",
                "R²",
                0.7650,
            ],
            [
                "Repair Cost",
                "Random Forest",
                "MAE (₹)",
                286.8328,
            ],
            [
                "Repair Cost",
                "Random Forest",
                "RMSE (₹)",
                572.4620,
            ],
            [
                "Repair Cost",
                "Random Forest",
                "R²",
                0.5540,
            ],
        ],
        columns=[
            "Task",
            "Deployment Model",
            "Metric",
            "Score",
        ],
    )

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Metric choice depends on the task: F1 for failure detection, "
        "Macro F1 for imbalanced failure-type classification, and MAE for "
        "RUL/cost regression. Accuracy is shown as a supporting metric."
    )

    st.markdown(
        "### 🚀 Deployment Model Stack"
    )

    stack = pd.DataFrame(
        [
            [
                "Failure Prediction",
                "MLP",
                "F1 Score",
                "best_failure_model.keras",
            ],
            [
                "Failure Type",
                "Balanced Extra Trees",
                "Macro F1",
                "best_failure_type_model.pkl",
            ],
            [
                "RUL",
                "XGBoost",
                "MAE",
                "best_rul_model.pkl",
            ],
            [
                "Repair Cost",
                "Random Forest",
                "MAE",
                "best_repair_cost_model.pkl",
            ],
        ],
        columns=[
            "Prediction Task",
            "Selected Model",
            "Primary Metric",
            "Deployment Artifact",
        ],
    )

    st.dataframe(
        stack,
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        "🏆 These four selected models are the actual models used by "
        "MaintainIQ predictions. The UI does not retrain models."
    )

def about():
    header(
        "ℹ️ About MaintainIQ",
        "Project overview, dataset, ML methodology and deployment.",
    )

    # ============================================================
    # PROJECT OVERVIEW
    # ============================================================

    st.markdown(
        """
        <div class="section-card">
            <div class="section-head">⚙️ Project Overview</div>
            <div style="
                color:#475467;
                font-size:13px;
                line-height:1.8;
                margin-top:9px;
            ">
                <b>MaintainIQ</b> is an AI-based predictive maintenance
                application that analyzes machine health and operating data
                to predict failures before they happen. It provides failure
                risk, failure type, Remaining Useful Life (RUL), and
                estimated repair-cost insights to support better maintenance
                decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='height:12px'></div>",
        unsafe_allow_html=True,
    )

    # ============================================================
    # PROJECT OBJECTIVE
    # ============================================================

    st.markdown(
        """
        <div class="section-card">
            <div class="section-head">🎯 Project Objective</div>
            <div style="
                color:#475467;
                font-size:13px;
                line-height:1.8;
                margin-top:9px;
            ">
                The goal of MaintainIQ is to reduce unexpected machine
                downtime and improve maintenance planning by providing
                early, data-driven insights about machine condition,
                possible failures, remaining machine life, and expected
                repair costs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='height:18px'></div>",
        unsafe_allow_html=True,
    )

    # ============================================================
    # DATASET & FEATURES
    # ============================================================

    st.markdown(
        "### 📊 Dataset & Features"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Dataset Records",
        "24,042",
    )

    c2.metric(
        "Training Records",
        "19,234",
    )

    c3.metric(
        "Test Records",
        "4,808",
    )

    c4.metric(
        "Processed Features",
        "19",
    )

    st.markdown(
        """
<div class="section-card" style="margin-top:12px">
    <b style="color:#14213d">14 model input features</b>
    <div style="color:#64748b;font-size:13px;line-height:1.8;margin-top:7px">
        machine_id, machine_type, vibration_rms, temperature_motor,
        current_phase_avg, pressure_level, rpm, operating_mode,
        hours_since_maintenance, ambient_temp, hour, day, month,
        day_of_week
    </div>
    <div style="color:#98a2b3;font-size:10px;margin-top:7px">
        Saved preprocessing performs numeric imputation/scaling and
        categorical imputation/one-hot encoding.
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )   
    # ============================================================
    # FINAL ML METHODOLOGY
    # ============================================================

    st.markdown(
        "### 🧠 Final ML Methodology"
    )

    method = pd.DataFrame(
        [
            [
                "Failure Prediction",
                "MLP",
                "Binary Classification",
                "F1 Score",
                "0.780890",
            ],
            [
                "Failure Type",
                "Balanced Extra Trees",
                "Multiclass Classification",
                "Macro F1",
                "0.8007",
            ],
            [
                "RUL",
                "XGBoost",
                "Regression",
                "MAE",
                "9.2303 hours",
            ],
            [
                "Repair Cost",
                "Random Forest",
                "Regression",
                "MAE",
                "₹286.83",
            ],
        ],
        columns=[
            "Task",
            "Final Model",
            "Problem Type",
            "Primary Metric",
            "Best Score",
        ],
    )

    st.dataframe(
        method,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "<div style='height:8px'></div>",
        unsafe_allow_html=True,
    )

    st.info(
        "ML architecture: binary classification + multiclass "
        "classification + regression + regression."
    )

    # ============================================================
    # DEPLOYMENT FLOW
    # ============================================================

    st.markdown(
        "### 🚀 Deployment Flow"
    )

    cols = st.columns(4)

    flow = [
        (
            "📡",
            "Machine Inputs",
            "Sensor + operating data",
        ),
        (
            "⚙️",
            "Preprocessor",
            "Saved feature pipeline",
        ),
        (
            "🤖",
            "ML Models",
            "MLP + Random Forest + XGBoost",
        ),
        (
            "📊",
            "Output",
            "Risk + type + RUL + cost",
        ),
    ]

    for col, item in zip(cols, flow):
        icon, title, text = item

        with col:
            st.markdown(
                f"""
<div class="section-card" style="min-height:135px">
    <div style="font-size:24px">{icon}</div>
    <div style="font-weight:800;color:#14213d;margin-top:7px">
        {title}
    </div>
    <div style="font-size:10px;color:#8a94a6;margin-top:4px">
        {text}
    </div>
</div>
                """,
                unsafe_allow_html=True,
            )
    # ============================================================
    # DEPLOYMENT ARTIFACTS
    # ============================================================

    st.markdown(
        "### 📦 Deployment Artifacts"
    )

    rows = []
    paths = model_paths()

    for key, filename in MODEL_FILES.items():
        rows.append(
            [
                filename,
                "✅ Available"
                if paths[key]
                else "❌ Missing",
            ]
        )

    st.dataframe(
        pd.DataFrame(
            rows,
            columns=[
                "Artifact",
                "Status",
            ],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Training and model selection are completed before deployment. "
        "The application loads the saved artifacts and performs inference."
    )

    # ============================================================
    # PROJECT QUOTE
    # ============================================================

    st.markdown(
        """
        <div style="
            margin-top:18px;
            padding:20px 24px;
            border-radius:18px;
            text-align:center;
            background:linear-gradient(135deg,#08172d,#123c59);
            color:white;
            box-shadow:0 10px 28px rgba(8,23,45,.12);
        ">
            <div style="
                font-size:15px;
                font-weight:800;
                letter-spacing:.2px;
            ">
                “Predict early. Maintain smart. Keep machines running.” ⚙️
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


PAGES = {
    "Home": home,
    "Dashboard": dashboard,
    "Predictions": predictions,
    "History": history,
    "Analytics": analytics,
    "Model Performance": performance,
    "About": about,
}

PAGES.get(
    st.session_state.page,
    home,
)()

st.markdown(
    """
    <div class="footer">
        ⚙️ MaintainIQ • AI-Powered Predictive Maintenance
        • Failure Detection • Failure Type • RUL • Repair Cost
    </div>
    """,
    unsafe_allow_html=True,
)
