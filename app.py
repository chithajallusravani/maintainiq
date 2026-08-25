import os

# ============================================================
# CPU-ONLY TENSORFLOW CONFIG
# Must be set BEFORE importing TensorFlow
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px




# ============================================================
# MAINTAINIQ - FINAL DEPLOYMENT APP
# Login: admin / maintainiq
# ============================================================

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
    "machine_id", "machine_type", "vibration_rms",
    "temperature_motor", "current_phase_avg", "pressure_level",
    "rpm", "operating_mode", "hours_since_maintenance",
    "ambient_temp", "hour", "day", "month", "day_of_week",
]

MACHINE_TYPES = ["CNC", "lathe", "milling", "press", "pump", "compressor"]
OPERATING_MODES = ["idle", "normal", "maintenance", "heavy_load"]

# Verified final deployment winners from the ML evaluation.
WINNERS = {
    "Failure Prediction": {
        "model": "MLP", "metric": "F1 Score", "score": 0.780890,
        "artifact": "best_failure_model.keras",
    },
    "Failure Type": {
        "model": "Balanced Extra Trees", "metric": "Macro F1", "score": 0.8007,
        "artifact": "best_failure_type_model.pkl",
    },
    "Remaining Useful Life": {
        "model": "XGBoost", "metric": "MAE", "score": 9.2303,
        "artifact": "best_rul_model.pkl",
    },
    "Estimated Repair Cost": {
        "model": "Random Forest", "metric": "MAE", "score": 286.8328,
        "artifact": "best_repair_cost_model.pkl",
    },
}

FALLBACKS = {
    "failure": pd.DataFrame([
        ["MLP", 0.780890, np.nan, np.nan, "🏆 Selected"],
    ], columns=["Model", "F1 Score", "Accuracy", "ROC-AUC", "Status"]),

    "failure_type": pd.DataFrame([
        ["Balanced Extra Trees", 0.8007, 0.9561, "🏆 Selected"],
        ["Balanced Subsample RF", 0.7142, 0.9507, ""],
        ["Current Random Forest", 0.6823, 0.9511, ""],
        ["Balanced Random Forest", 0.6729, 0.9509, ""],
    ], columns=["Model", "Macro F1", "Accuracy", "Status"]),

    "rul": pd.DataFrame([
        ["XGBoost", 9.2303, 13.2020, 0.7650, "🏆 Selected"],
    ], columns=["Model", "MAE (hours)", "RMSE (hours)", "R²", "Status"]),

    "repair": pd.DataFrame([
        ["Random Forest", 286.8328, 572.4620, 0.5540, "🏆 Selected"],
    ], columns=["Model", "MAE (₹)", "RMSE (₹)", "R²", "Status"]),
}

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# UI
# ============================================================

st.markdown("""
<style>
.stApp{
    background:
    radial-gradient(circle at 8% 4%,rgba(59,130,246,.10),transparent 26%),
    radial-gradient(circle at 92% 2%,rgba(139,92,246,.10),transparent 25%),
    #f5f7fb;
}
.block-container{max-width:1500px;padding-top:1.2rem;padding-bottom:2rem}
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0b1220 0%,#111b31 52%,#172554 100%);
}
[data-testid="stSidebar"] .stButton>button{
    width:100%;min-height:43px;border-radius:12px;
    background:rgba(255,255,255,.035);
    border:1px solid rgba(255,255,255,.08);
    color:#e5edff;font-weight:700;text-align:left;
}
[data-testid="stSidebar"] .stButton>button:hover{
    background:linear-gradient(90deg,rgba(37,99,235,.30),rgba(124,58,237,.25));
    color:white;
}
.page-title{font-size:34px;font-weight:850;color:#10234f;letter-spacing:-1px}
.page-subtitle{font-size:15px;color:#64748b;margin-bottom:20px}
.hero{
    padding:30px;border-radius:24px;color:white;
    background:linear-gradient(135deg,#2563eb,#4f46e5 55%,#8b5cf6);
    box-shadow:0 18px 45px rgba(37,99,235,.22);
}
.hero h1{margin:0 0 8px;font-size:32px}
.hero p{margin:0;color:rgba(255,255,255,.90);line-height:1.65}
.card{
    background:white;border:1px solid #e2e8f0;border-radius:20px;
    padding:22px;box-shadow:0 8px 28px rgba(15,23,42,.06);
}
.winner{
    padding:14px;border-radius:15px;margin-bottom:9px;
    border:1px solid #dbeafe;
    background:linear-gradient(135deg,#eff6ff,#f5f3ff);
}
.small{color:#64748b;font-size:12px}
.pred{
    background:white;border:1px solid #e2e8f0;border-radius:20px;
    padding:20px;min-height:155px;
    box-shadow:0 8px 25px rgba(15,23,42,.06);
}
.pred-icon{font-size:27px}
.pred-label{color:#64748b;font-size:12px;font-weight:800;margin-top:8px}
.pred-value{color:#0f172a;font-size:25px;font-weight:850;margin-top:4px}
.badge{display:inline-block;margin-top:8px;padding:5px 10px;border-radius:18px;font-size:11px;font-weight:800}
.green{background:#d1fae5;color:#047857}
.orange{background:#fef3c7;color:#b45309}
.red{background:#fee2e2;color:#b91c1c}
.blue{background:#dbeafe;color:#1d4ed8}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS + MODELS
# ============================================================

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
    # Import TensorFlow only when ML models are actually required.
    from tensorflow.keras.models import load_model

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


# ============================================================
# SESSION
# ============================================================

for key, default in {
    "authenticated": False,
    "page": "Home",
    "history": [],
    "last_prediction": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# LOGIN
# ============================================================

def login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.15, 1])

    with center:
        st.markdown("""
        <div class="card" style="text-align:center">
            <div style="
                width:68px;height:68px;margin:auto;border-radius:20px;
                display:flex;align-items:center;justify-content:center;
                background:linear-gradient(135deg,#3b82f6,#8b5cf6);
                color:white;font-size:34px;">⚙️</div>
            <div style="font-size:31px;font-weight:850;color:#10234f;margin-top:10px">
                MaintainIQ
            </div>
            <div style="color:#64748b;font-size:13px">
                Intelligent Predictive Maintenance
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container():
            st.markdown("### 🔐 Sign in")
            st.caption("Access your predictive maintenance control center.")
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password",
                                      placeholder="Enter password")

            if st.button("🚀 Sign In", type="primary",
                         use_container_width=True):
                if username.strip().lower() == APP_USERNAME and password == APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.session_state.page = "Home"
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

            st.info("Demo login: admin / maintainiq")


if not st.session_state.authenticated:
    login_page()
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:6px 5px 16px">
        <div style="
            width:60px;height:60px;margin:auto;border-radius:18px;
            display:flex;align-items:center;justify-content:center;
            background:linear-gradient(135deg,#3b82f6,#8b5cf6);
            color:white;font-size:30px;">⚙️</div>
        <div style="color:white;font-size:25px;font-weight:850;margin-top:8px">
            MaintainIQ
        </div>
        <div style="color:#9fb3d8;font-size:12px">
            Predictive Maintenance
        </div>
    </div>

    <div style="
        padding:16px;border-radius:18px;background:rgba(255,255,255,.08);
        border:1px solid rgba(255,255,255,.10);margin-bottom:18px">
        <div style="display:flex;gap:11px;align-items:center">
            <div style="
                width:48px;height:48px;border-radius:14px;
                background:linear-gradient(135deg,#dbeafe,#ede9fe);
                display:flex;align-items:center;justify-content:center;
                font-size:24px">👨‍💼</div>
            <div>
                <div style="color:white;font-weight:800">Hello, Admin</div>
                <div style="color:#9fb3d8;font-size:11px">Administrator</div>
            </div>
        </div>
        <div style="
            display:inline-block;margin-top:12px;padding:5px 10px;
            border-radius:20px;background:rgba(34,197,94,.12);
            border:1px solid rgba(34,197,94,.30);
            color:#4ade80;font-size:11px;font-weight:700">
            ● Online
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="color:#7183a8;font-size:10px;font-weight:800;'
        'letter-spacing:1px;margin:0 4px 8px">MAIN MENU</div>',
        unsafe_allow_html=True,
    )

    nav = [
        ("🏠", "Home"), ("📊", "Dashboard"), ("🔮", "Predictions"),
        ("📜", "History"), ("📈", "Analytics"),
        ("🤖", "Model Performance"), ("ℹ️", "About"),
    ]

    for icon, label in nav:
        if st.button(f"{icon}   {label}", key=f"nav_{label}",
                     use_container_width=True):
            st.session_state.page = label
            st.rerun()

    if st.button("🚪   Logout", key="logout",
                 use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.page = "Home"
        st.session_state.history = []
        st.session_state.last_prediction = None
        st.rerun()


# ============================================================
# LOAD
# ============================================================

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


# ============================================================
# PREPROCESSOR FIX
# ============================================================

def expected_features():
    pre = models["preprocessor"]

    names = getattr(pre, "feature_names_in_", None)
    if names is not None:
        return [str(x) for x in names]

    found = []
    try:
        for _, _, columns in pre.transformers_:
            if isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
                found.extend(str(x) for x in columns)
    except Exception:
        pass

    return list(dict.fromkeys(found)) if found else MODEL_INPUT_FEATURES.copy()


def prepare_input(
    machine_id, machine_type, vibration_rms, temperature_motor,
    current_phase_avg, pressure_level, rpm, operating_mode,
    hours_since_maintenance, ambient_temp, prediction_dt
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

    # Critical fix:
    # current_phase_avg and pressure_level are explicitly included here.
    return pd.DataFrame(
        [[values[x] for x in expected]],
        columns=expected,
    )


# ============================================================
# PREDICTION
# ============================================================

def prediction(input_df):
    processed = models["preprocessor"].transform(input_df)

    # Failure probability
    raw = np.asarray(models["failure"].predict(processed, verbose=0))
    if raw.ndim == 2 and raw.shape[1] == 2:
        probability = float(raw[0, 1])
    else:
        probability = float(raw.reshape(-1)[0])
    probability = float(np.clip(probability, 0, 1))

    failure_decision = "Failure Risk" if probability >= 0.50 else "No Failure"

    # Failure type
    raw_type = np.asarray(models["failure_type"].predict(processed)).reshape(-1)[0]
    if isinstance(raw_type, str):
        failure_type = raw_type
    else:
        failure_type = str(
            models["encoder"].inverse_transform([int(raw_type)])[0]
        )
    failure_type = failure_type.strip().lower()

    # RUL
    rul = float(np.asarray(models["rul"].predict(processed)).reshape(-1)[0])
    rul = max(0.0, rul)

    # Repair cost
    cost = float(
        np.asarray(models["repair_cost"].predict(processed)).reshape(-1)[0]
    )
    cost = max(0.0, cost)

    if probability >= 0.50:
        risk, risk_class = "High", "red"
    elif probability >= 0.20:
        risk, risk_class = "Moderate", "orange"
    else:
        risk, risk_class = "Low", "green"

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "machine_id": int(input_df.iloc[0]["machine_id"]),
        "machine_type": str(input_df.iloc[0]["machine_type"]),
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
        ("⚠️", "Failure Probability",
         f'{result["failure_probability"]*100:.2f}%',
         f'{result["risk_level"]} Risk', result["risk_class"]),
        ("🔧", "Failure Type",
         result["failure_type"].replace("_", " ").title(),
         "Model classified", "blue"),
        ("⏱️", "Remaining Useful Life",
         f'{result["rul_hours"]:.2f} h',
         "Estimated remaining life", "blue"),
        ("💰", "Estimated Repair Cost",
         f'₹{result["repair_cost"]:,.2f}',
         "Model estimate", "green"),
    ]

    for col, (icon, label, value, note, color) in zip(cols, data):
        with col:
            st.markdown(f"""
            <div class="pred">
                <div class="pred-icon">{icon}</div>
                <div class="pred-label">{label}</div>
                <div class="pred-value">{value}</div>
                <span class="badge {color}">{note}</span>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# PAGE HELPERS
# ============================================================

def header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>',
                unsafe_allow_html=True)


# ============================================================
# HOME
# ============================================================

def home():
    st.markdown("""
    <div class="hero">
        <h1>Welcome to MaintainIQ 👋</h1>
        <p>
            Intelligent predictive maintenance for smarter machine operations.
            Predict failure risk, failure type, remaining useful life and
            estimated repair cost from machine conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    h = st.session_state.history
    total = len(h)
    avg_risk = np.mean([x["failure_probability"] for x in h])*100 if h else 0
    avg_rul = np.mean([x["rul_hours"] for x in h]) if h else 0
    high = sum(x["failure_probability"] >= .50 for x in h)
    total_cost = sum(x["repair_cost"] for x in h)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predictions", total)
    c2.metric("High Risk", high)
    c3.metric("Average Risk", f"{avg_risk:.2f}%")
    c4.metric("Estimated Cost", f"₹{total_cost:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    a, b = st.columns([1.3, 1])

    with a:
        with st.container():
            st.markdown("### 🚀 What MaintainIQ Does")
            st.write(
                "MaintainIQ converts machine-health and operating information "
                "into practical predictive-maintenance insights."
            )
            st.markdown(
                "- ⚠️ **Failure prediction** within 24 hours\n"
                "- 🔧 **Failure type** classification\n"
                "- ⏱️ **RUL** estimation in hours\n"
                "- 💰 **Repair cost** estimation"
            )
            st.success("All six deployment artifacts are loaded.")

    with b:
        with st.container():
            st.markdown("### 🏆 Final Deployment Winners")
            for task, info in WINNERS.items():
                st.markdown(f"""
                <div class="winner">
                    <div class="small">{task}</div>
                    <b>🏆 {info["model"]}</b><br>
                    <span class="small">
                        {info["metric"]}: {info["score"]}
                    </span>
                </div>
                """, unsafe_allow_html=True)

    x, y = st.columns(2)
    with x:
        if st.button("🔮 Make New Prediction", type="primary",
                     use_container_width=True):
            st.session_state.page = "Predictions"
            st.rerun()
    with y:
        if st.button("📊 Open Dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
            st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

def dashboard():
    header("📊 Dashboard",
           "Quick view of prediction activity generated in this session.")

    if not st.session_state.history:
        st.info("No predictions yet. Generate a prediction first.")
        return

    df = pd.DataFrame(st.session_state.history)
    high = int((df.failure_probability >= .50).sum())
    moderate = int(((df.failure_probability >= .20) &
                    (df.failure_probability < .50)).sum())
    low = len(df) - high - moderate

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(df))
    c2.metric("High Risk", high)
    c3.metric("Moderate", moderate)
    c4.metric("Low Risk", low)

    a, b = st.columns(2)

    with a:
        risk = pd.DataFrame({
            "Risk": ["Low", "Moderate", "High"],
            "Machines": [low, moderate, high],
        })
        fig = px.pie(
            risk, names="Risk", values="Machines", hole=.55,
            color="Risk",
            color_discrete_map={
                "Low": "#10b981",
                "Moderate": "#f59e0b",
                "High": "#ef4444",
            },
            title="Risk Distribution",
        )
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)

    with b:
        top = df.sort_values("repair_cost", ascending=False).head(10)
        fig = px.bar(
            top, x="machine_id", y="repair_cost",
            color="repair_cost",
            color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
            title="Top Estimated Repair Costs",
            labels={"machine_id":"Machine ID",
                    "repair_cost":"Repair Cost (₹)"},
        )
        fig.update_layout(height=390, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧾 Recent Prediction Activity")
    view = df.copy()
    view["Failure Probability"] = (
        view.failure_probability*100).round(2).astype(str) + "%"
    view["Failure Type"] = (
        view.failure_type.str.replace("_"," ",regex=False).str.title()
    )
    view = view[[
        "timestamp","machine_id","machine_type",
        "Failure Probability","failure_prediction",
        "Failure Type","risk_level","rul_hours","repair_cost"
    ]].rename(columns={
        "timestamp":"Time","machine_id":"Machine ID",
        "machine_type":"Machine Type",
        "failure_prediction":"Prediction","risk_level":"Risk",
        "rul_hours":"RUL (Hours)","repair_cost":"Repair Cost (₹)",
    })
    st.dataframe(view, use_container_width=True, hide_index=True)


# ============================================================
# PREDICTIONS
# ============================================================

def predictions():
    header("🔮 Predictions",
           "Enter current machine conditions and generate four ML predictions.")

    st.info(
        "Pipeline: machine inputs → saved preprocessor → selected deployment "
        "models → risk, type, RUL and repair-cost predictions."
    )

    with st.form("prediction_form"):
        st.markdown("### ⚙️ Machine Information")
        a, b, c = st.columns(3)

        with a:
            machine_id = st.number_input("Machine ID", min_value=1,
                                         value=941, step=1)
            machine_type = st.selectbox("Machine Type", MACHINE_TYPES)

        with b:
            operating_mode = st.selectbox("Operating Mode", OPERATING_MODES)
            hours_since = st.number_input(
                "Hours Since Maintenance", min_value=0.0,
                value=147.12, step=1.0
            )

        with c:
            pdate = st.date_input("Prediction Date", datetime.now().date())
            ptime = st.time_input(
                "Prediction Time",
                datetime.now().time().replace(second=0, microsecond=0)
            )

        st.markdown("### 📡 Sensor Measurements")
        a, b, c = st.columns(3)

        with a:
            vibration = st.number_input("Vibration RMS", min_value=0.0,
                                        value=.83, step=.01)
            temperature = st.number_input("Motor Temperature",
                                          value=47.85, step=.1)

        with b:
            current = st.number_input("Average Phase Current",
                                      min_value=0.0, value=4.29, step=.01)
            pressure = st.number_input("Pressure Level",
                                       min_value=0.0, value=24.90, step=.1)

        with c:
            rpm = st.number_input("RPM", min_value=0.0,
                                  value=895.4, step=.1)
            ambient = st.number_input("Ambient Temperature",
                                      value=10.5, step=.1)

        submitted = st.form_submit_button(
            "🔮 Generate Predictions",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        dt = datetime.combine(pdate, ptime)
        try:
            X = prepare_input(
                machine_id, machine_type, vibration, temperature,
                current, pressure, rpm, operating_mode,
                hours_since, ambient, dt
            )

            with st.spinner("Running MaintainIQ ML pipeline..."):
                result = prediction(X)

            save_result(result)
            st.success("Prediction completed successfully.")
            cards(result)

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

            st.markdown("### 💡 Maintenance Insight")
            if result["risk_level"] == "High":
                text = (
                    f"Prioritize inspection for the predicted "
                    f"{result['failure_type'].replace('_',' ')} condition."
                )
            elif result["risk_level"] == "Moderate":
                text = "Continue close monitoring and plan preventive maintenance."
            else:
                text = "Machine is currently in the low-risk range."

            st.info(
                text + f" Estimated RUL: {result['rul_hours']:.2f} hours."
            )

            with st.expander("🔍 View exact model input"):
                st.dataframe(X, use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error("Prediction failed.")
            st.exception(exc)

    elif st.session_state.last_prediction:
        st.markdown("### Latest Prediction")
        cards(st.session_state.last_prediction)


# ============================================================
# HISTORY
# ============================================================

def history():
    header("📜 Prediction History",
           "Predictions generated during the current application session.")

    if not st.session_state.history:
        st.info("No history yet. Generate a prediction first.")
        return

    df = pd.DataFrame(st.session_state.history)

    a, b = st.columns(2)
    with a:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_prediction = None
            st.rerun()
    with b:
        st.download_button(
            "⬇️ Export Prediction History",
            df.to_csv(index=False).encode("utf-8"),
            file_name="maintainiq_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    view = df.copy()
    view["failure_probability"] = (
        view.failure_probability*100).round(2).astype(str)+"%"
    view["failure_type"] = (
        view.failure_type.str.replace("_"," ",regex=False).str.title()
    )
    view = view.rename(columns={
        "timestamp":"Time","machine_id":"Machine ID",
        "machine_type":"Machine Type",
        "failure_probability":"Failure Probability",
        "failure_prediction":"Prediction",
        "failure_type":"Failure Type",
        "risk_level":"Risk",
        "rul_hours":"RUL (Hours)",
        "repair_cost":"Repair Cost (₹)",
    })
    st.dataframe(view, use_container_width=True, hide_index=True)


# ============================================================
# ANALYTICS
# ============================================================

def analytics():
    header("📈 Analytics",
           "Colorful visual analytics from predictions in this session.")

    if not st.session_state.history:
        st.info("Generate a few predictions first.")
        return

    df = pd.DataFrame(st.session_state.history)
    a, b = st.columns(2)

    with a:
        fig = px.histogram(
            df, x="failure_probability", nbins=10,
            color_discrete_sequence=["#6366f1"],
            title="Failure Probability Distribution",
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with b:
        fig = px.scatter(
            df, x="rul_hours", y="repair_cost",
            color="risk_level", size="failure_probability",
            hover_data=["machine_id","failure_type"],
            color_discrete_map={
                "Low":"#10b981","Moderate":"#f59e0b","High":"#ef4444"
            },
            title="RUL vs Estimated Repair Cost",
            labels={"rul_hours":"RUL (hours)",
                    "repair_cost":"Repair Cost (₹)",
                    "risk_level":"Risk"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    counts = df.failure_type.value_counts().reset_index()
    counts.columns = ["Failure Type","Count"]

    fig = px.bar(
        counts, x="Failure Type", y="Count", color="Failure Type",
        title="Predicted Failure Types",
        color_discrete_sequence=[
            "#3b82f6","#8b5cf6","#ec4899","#f59e0b","#10b981"
        ],
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "These charts describe current-session predictions, not training/test "
        "evaluation. Official metrics are shown in Model Performance."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def comparison(filename, fallback, selected):
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
    header("🤖 Model Performance",
           "Candidate comparison, evaluation metrics and final deployment winners.")

    st.success(
        "Each ML task was evaluated independently. The best task-specific "
        "model was selected for deployment."
    )

    st.markdown("### 🏆 Final Deployment Winners")
    cols = st.columns(4)

    for col, (task, info) in zip(cols, WINNERS.items()):
        with col:
            st.markdown(f"""
            <div class="winner">
                <div class="small">{task}</div>
                <b>🏆 {info["model"]}</b><br>
                <span class="small">{info["metric"]}: {info["score"]}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 1️⃣ Failure Prediction")
    st.dataframe(
        comparison("failure_model_comparison.csv",
                   FALLBACKS["failure"], "MLP"),
        use_container_width=True, hide_index=True
    )
    st.caption("Primary metric: F1 Score (higher is better). Deployment: MLP.")

    st.markdown("### 2️⃣ Failure Type Classification")
    st.dataframe(
        comparison("failure_type_model_comparison.csv",
                   FALLBACKS["failure_type"], "Balanced Extra Trees"),
        use_container_width=True, hide_index=True
    )
    st.caption(
        "Primary metric: Macro F1 (higher is better). "
        "Deployment: Balanced Extra Trees."
    )

    st.markdown("### 3️⃣ Remaining Useful Life")
    st.dataframe(
        comparison("rul_model_comparison.csv",
                   FALLBACKS["rul"], "XGBoost"),
        use_container_width=True, hide_index=True
    )
    st.caption("Primary metric: MAE (lower is better). Deployment: XGBoost.")

    st.markdown("### 4️⃣ Estimated Repair Cost")
    st.dataframe(
        comparison("repair_cost_model_comparison.csv",
                   FALLBACKS["repair"], "Random Forest"),
        use_container_width=True, hide_index=True
    )
    st.caption(
        "Primary metric: MAE (lower is better). Deployment: Random Forest."
    )

    st.markdown("### 📊 Key Evaluation Metrics")
    metrics = pd.DataFrame([
        ["Failure Prediction","MLP","F1 Score",0.780890],
        ["Failure Type","Balanced Extra Trees","Macro F1",0.8007],
        ["Failure Type","Balanced Extra Trees","Accuracy",0.9561],
        ["RUL","XGBoost","MAE (hours)",9.2303],
        ["RUL","XGBoost","RMSE (hours)",13.2020],
        ["RUL","XGBoost","R²",0.7650],
        ["Repair Cost","Random Forest","MAE (₹)",286.8328],
        ["Repair Cost","Random Forest","RMSE (₹)",572.4620],
        ["Repair Cost","Random Forest","R²",0.5540],
    ], columns=["Task","Deployment Model","Metric","Score"])
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.info(
        "Metric choice depends on the task: F1 for failure detection, "
        "Macro F1 for imbalanced failure-type classification, and MAE for "
        "RUL/cost regression. Accuracy is shown as a supporting metric."
    )

    st.markdown("### 🚀 Deployment Model Stack")
    stack = pd.DataFrame([
        ["Failure Prediction","MLP","F1 Score","best_failure_model.keras"],
        ["Failure Type","Balanced Extra Trees","Macro F1",
         "best_failure_type_model.pkl"],
        ["RUL","XGBoost","MAE","best_rul_model.pkl"],
        ["Repair Cost","Random Forest","MAE","best_repair_cost_model.pkl"],
    ], columns=[
        "Prediction Task","Selected Model","Primary Metric",
        "Deployment Artifact"
    ])
    st.dataframe(stack, use_container_width=True, hide_index=True)

    st.success(
        "🏆 These four selected models are the actual models used by "
        "MaintainIQ predictions. The UI does not retrain models."
    )


# ============================================================
# ABOUT
# ============================================================

def about():
    header("ℹ️ About MaintainIQ",
           "Project overview, dataset, ML methodology and deployment.")

    a, b = st.columns(2)

    with a:
        with st.container():
            st.markdown("### ⚙️ Project Overview")
            st.write(
                "**MaintainIQ** is an intelligent predictive-maintenance "
                "application designed to predict machine risk before unexpected "
                "failures occur."
            )
            st.write("The system produces four outputs:")
            st.markdown(
                "- ⚠️ Failure probability within 24 hours\n"
                "- 🔧 Predicted failure type\n"
                "- ⏱️ Remaining Useful Life (RUL)\n"
                "- 💰 Estimated repair cost"
            )

    with b:
        with st.container():
            st.markdown("### 🎯 Project Objective")
            st.write(
                "Convert machine telemetry into actionable maintenance "
                "intelligence so maintenance teams can identify risk, understand "
                "the likely failure mode, estimate remaining machine life and "
                "prepare an approximate repair budget."
            )
            st.success(
                "ML architecture: binary classification + multiclass "
                "classification + regression + regression."
            )

    st.markdown("### 📊 Dataset & Features")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records","24,042")
    c2.metric("Training Records","19,234")
    c3.metric("Test Records","4,808")
    c4.metric("Processed Features","19")

    with st.container():
        st.markdown("**14 model input features**")
        st.write(
            "machine_id, machine_type, vibration_rms, temperature_motor, "
            "current_phase_avg, pressure_level, rpm, operating_mode, "
            "hours_since_maintenance, ambient_temp, hour, day, month, "
            "day_of_week"
        )
        st.caption(
            "Saved preprocessing performs numeric imputation/scaling and "
            "categorical imputation/one-hot encoding."
        )

    st.markdown("### 🧠 Final ML Methodology")
    method = pd.DataFrame([
        ["Failure Prediction","MLP","Binary Classification","F1 Score","0.780890"],
        ["Failure Type","Balanced Extra Trees","Multiclass Classification",
         "Macro F1","0.8007"],
        ["RUL","XGBoost","Regression","MAE","9.2303 hours"],
        ["Repair Cost","Random Forest","Regression","MAE","₹286.83"],
    ], columns=["Task","Final Model","Problem Type","Primary Metric","Best Score"])
    st.dataframe(method, use_container_width=True, hide_index=True)

    st.markdown("### 🚀 Deployment Flow")
    cols = st.columns(4)
    flow = [
        ("📡","Machine Inputs","Sensor + operating data"),
        ("⚙️","Preprocessor","Saved feature pipeline"),
        ("🤖","ML Models","MLP + Random Forest + XGBoost"),
        ("📊","Output","Risk + type + RUL + cost"),
    ]
    for col, (icon,title,text) in zip(cols,flow):
        with col:
            with st.container():
                st.markdown(f"### {icon}")
                st.markdown(f"**{title}**")
                st.caption(text)

    st.markdown("### 📦 Deployment Artifacts")
    rows = []
    paths = model_paths()
    for key, filename in MODEL_FILES.items():
        rows.append([filename, "✅ Available" if paths[key] else "❌ Missing"])
    st.dataframe(
        pd.DataFrame(rows, columns=["Artifact","Status"]),
        use_container_width=True, hide_index=True
    )

    st.info(
        "Training and model selection are completed before deployment. "
        "The Streamlit application loads the saved artifacts and performs inference."
    )


# ============================================================
# ROUTER
# ============================================================

PAGES = {
    "Home": home,
    "Dashboard": dashboard,
    "Predictions": predictions,
    "History": history,
    "Analytics": analytics,
    "Model Performance": performance,
    "About": about,
}

PAGES.get(st.session_state.page, home)()

st.markdown("""
<div style="
    text-align:center;color:#94a3b8;font-size:11px;padding:25px 0 5px">
    ⚙️ MaintainIQ • AI-Powered Predictive Maintenance
    • Failure Detection • Failure Type • RUL • Repair Cost
</div>
""", unsafe_allow_html=True)
