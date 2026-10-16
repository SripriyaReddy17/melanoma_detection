import streamlit as st
import requests
import sqlite3
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from PIL import Image

from frontend_helpers import (
    classify_persistence_status,
    should_save_to_history,
    parse_api_error,
    build_temp_case,
    generate_report_export,
)

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="CEFM Clinical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000/api/v1/analyze"
HEALTH_URL = "http://localhost:8000/api/v1/health"
DB_PATH = Path(__file__).parent / "cefm_frontend.db"

BODY_AREAS = [
    "Scalp", "Face", "Ear", "Neck", "Chest", "Abdomen", "Back",
    "Shoulder", "Upper arm", "Forearm", "Hand", "Thigh", "Leg", "Foot", "Other"
]

# =========================================================
# DESIGN
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  color: #0f172a;
}
body, .stApp {
    background: #f1f5f9;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1140px; }

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
  border-right: 1px solid #1f2937;
}
section[data-testid="stSidebar"] * { color: #e5e7eb !important; }
section[data-testid="stSidebar"] .stRadio label {
  padding: 0.7rem 0.85rem;
  border-radius: 10px;
}

.hero {
  background: linear-gradient(145deg, #0b1220 0%, #1e293b 100%);
  border: 1px solid #1f2937;
  border-radius: 18px;
  padding: 1.45rem 1.6rem;
  margin-bottom: 1.15rem;
}
.hero h1 {
  margin: 0;
  color: #fff;
  font-size: 1.48rem;
  font-weight: 700;
  letter-spacing: -0.02em;
}
.hero p {
  margin: 0.42rem 0 0;
  color: #94a3b8;
  font-size: 0.94rem;
  line-height: 1.5;
}

.section {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin: 1.25rem 0 0.7rem;
  font-size: 0.95rem;
  font-weight: 700;
  color: #0f172a;
}
.section:after {
  content: "";
  flex: 1;
  height: 1px;
  background: #e2e8f0;
}
.badge-step {
  width: 22px; height: 22px; border-radius: 999px;
  background: #0f172a; color: #fff;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 0.72rem; font-weight: 700;
}

.card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 1rem 1.1rem;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  margin-bottom: 0.8rem;
}
.card-label {
  font-size: 0.72rem;
  font-weight: 650;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: #64748b;
}
.card-value {
  margin-top: 0.28rem;
  font-size: 1.12rem;
  font-weight: 700;
  color: #0f172a;
  line-height: 1.3;
}
.card-sub {
  margin-top: 0.22rem;
  font-size: 0.85rem;
  color: #64748b;
}

.pill {
  display: inline-block;
  padding: 0.2rem 0.68rem;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 650;
}
.pill-red { background:#fef2f2; color:#b91c1c; border:1px solid #fecaca; }
.pill-amber { background:#fffbeb; color:#b45309; border:1px solid #fde68a; }
.pill-green { background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0; }
.pill-slate { background:#f8fafc; color:#475569; border:1px solid #e2e8f0; }

.box {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1rem 1.1rem;
  color: #334155;
  line-height: 1.6;
  white-space: pre-wrap;
  font-size: 0.93rem;
}
.box-accent { border-left: 4px solid #0f172a; }

.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
  border-radius: 10px !important;
}
div[data-testid="stFileUploader"] {
  background: #f8fafc;
  border: 1px dashed #cbd5e1;
  border-radius: 12px;
  padding: 0.55rem;
}
.stButton > button[kind="primary"] {
  background: #0f172a;
  border: 1px solid #0f172a;
  border-radius: 10px;
  height: 2.7rem;
  font-weight: 650;
}
.stButton > button[kind="primary"]:hover {
  background: #1e293b;
  border-color: #1e293b;
}
.auth-wrap { max-width: 460px; margin: 0.4rem auto 0; }
.muted { color: #64748b; font-size: 0.86rem; }
</style>
""", unsafe_allow_html=True)

# =========================================================
#  Frontend DATABASE
# =========================================================
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            full_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_user_id INTEGER NOT NULL,
            patient_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            visit_date TEXT,
            lesion_area TEXT,
            lesion_size TEXT,
            duration TEXT,
            symptoms TEXT,
            explanation TEXT,
            image_name TEXT,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def register_user(email, password, role, full_name):
    conn = db()
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash, role, full_name, created_at) VALUES (?,?,?,?,?)",
            (email.lower().strip(), hash_pw(password), role, full_name.strip(), datetime.now().isoformat())
        )
        conn.commit()
        return True, "Account created. Please login."
    except sqlite3.IntegrityError:
        return False, "Email already registered."
    finally:
        conn.close()

def login_user(email, password):
    conn = db()
    row = conn.execute(
        "SELECT * FROM users WHERE email=? AND password_hash=?",
        (email.lower().strip(), hash_pw(password))
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def save_case(user_id, patient, lesion, image_name, result):
    conn = db()
    conn.execute("""
        INSERT INTO cases (
            patient_user_id, patient_name, age, gender, phone, visit_date,
            lesion_area, lesion_size, duration, symptoms, explanation,
            image_name, result_json, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id,
        patient["full_name"], patient["age"], patient["gender"], patient.get("phone", ""),
        patient["visit_date"], lesion["area"], lesion.get("size", ""), lesion.get("duration", ""),
        json.dumps(lesion.get("symptoms", [])), lesion["explanation"], image_name,
        json.dumps(result), datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

def cases_for_patient(user_id):
    conn = db()
    rows = conn.execute(
        "SELECT * FROM cases WHERE patient_user_id=? ORDER BY id DESC", (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def all_cases():
    conn = db()
    rows = conn.execute("SELECT * FROM cases ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_case(case_id):
    conn = db()
    row = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def api_online():
    try:
        return requests.get(HEALTH_URL, timeout=3).status_code == 200
    except Exception:
        return False

def pill(text, kind="slate"):
    return f'<span class="pill pill-{kind}">{text}</span>'

def sev_kind(v):
    v = (v or "").lower()
    if v in ["high", "severe"]:
        return "red"
    if v == "medium":
        return "amber"
    return "green"

init_db()

if "user" not in st.session_state:
    st.session_state.user = None
if "unsaved_case" not in st.session_state:
    st.session_state.unsaved_case = None
if "unsaved_storage_warning" not in st.session_state:
    st.session_state.unsaved_storage_warning = None

# =========================================================
# AUTH   //session+login/register
# =========================================================
def auth_view():
    st.markdown("""
    <div class="hero">
      <h1>CEFM Clinical Assistant</h1>
      <p>Explainable lesion assessment with patient history and doctor review.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        email = st.text_input("Email", key="login_email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", key="login_pw")
        if st.button("Sign in", type="primary"):
            user = login_user(email, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid email or password")

    with tab_register:
        name = st.text_input("Full name", key="reg_name")
        email = st.text_input("Email", key="reg_email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", key="reg_pw")
        role = st.selectbox("Register as", ["patient", "doctor"])
        if st.button("Create account", type="primary"):
            if not name.strip() or not email.strip() or not password.strip():
                st.error("Please fill all fields")
            else:
                ok, msg = register_user(email, password, role, name)
                st.success(msg) if ok else st.error(msg)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="muted" style="text-align:center;margin-top:1rem;">For research and clinical decision support only.</p>',
        unsafe_allow_html=True
    )

# =========================================================
# SHARED CASE FORM + ANALYZE
# =========================================================
def case_form_and_analyze(submit_label, notes_label, success_msg, save_user_id):
    if not api_online():
        st.warning("Start backend first: `uvicorn app.main:app --host 0.0.0.0 --port 8000`")
        st.stop()

    if st.session_state.get("unsaved_case"):
        unsaved = st.session_state.unsaved_case
        with st.expander("📌 Active Unsaved Analysis from Current Session", expanded=False):
            st.warning(
                "⚠️ " + (
                    st.session_state.get("unsaved_storage_warning")
                    or "Analysis completed, but not saved to server database history."
                )
            )
            c_top1, c_top2 = st.columns([1, 1])
            with c_top1:
                st.download_button(
                    "📥 Download Unsaved Report (.md)",
                    data=generate_report_export(unsaved),
                    file_name=f"CEFM_Unsaved_{unsaved['patient_name']}_{datetime.now():%Y%m%d_%H%M}.md",
                    mime="text/markdown",
                    key="top_unsaved_dl_md"
                )
            with c_top2:
                if st.button("Discard Unsaved Case", key="top_discard_unsaved_btn"):
                    st.session_state.unsaved_case = None
                    st.session_state.unsaved_storage_warning = None
                    st.rerun()
            render_report(unsaved)

    st.markdown('<div class="section"><span class="badge-step">1</span> Patient information</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1.45, 0.7, 0.85, 1])
    with c1:
        default_name = st.session_state.user["full_name"] if st.session_state.user["role"] == "patient" else ""
        full_name = st.text_input("Full name *", value=default_name)
    with c2:
        age = st.number_input("Age *", 1, 120, 35)
    with c3:
        gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
    with c4:
        visit_date = st.date_input("Visit date", value=date.today())
    phone = st.text_input("Phone")

    st.markdown('<div class="section"><span class="badge-step">2</span> Lesion details</div>', unsafe_allow_html=True)
    l1, l2, l3 = st.columns(3)
    with l1:
        lesion_area = st.selectbox("Area of lesion / body site *", BODY_AREAS)
    with l2:
        lesion_size = st.text_input("Patient-reported approximate size (optional)", placeholder="e.g. 4 mm")
    with l3:
        duration = st.text_input("Duration noticed", placeholder="e.g. 3 months")

    symptoms = st.multiselect(
        "Symptoms",
        ["None", "Itching", "Bleeding", "Pain", "Color change", "Size increase", "Ulceration"]
    )
    explanation = st.text_area(notes_label, height=115,
                               placeholder="Describe lesion appearance, changes over time, and concerns...")

    st.markdown('<div class="section"><span class="badge-step">3</span> Skin lesion image</div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        uploaded = st.file_uploader("Upload image *", type=["jpg", "jpeg", "png"])
        st.markdown("""
        <div class="box box-accent">
        <b>Before taking the photograph</b><br>
        • Place either the provided <b>20 mm × 20 mm calibration marker</b> or a ruler with clear <b>1 mm graduations</b> beside the lesion.<br>
        • Keep the calibration reference and lesion on the same skin surface.<br>
        • Keep the camera approximately parallel to the skin.<br>
        • Make the complete lesion and complete marker visible.<br>
        • Use good, even lighting; avoid glare, shadows and blur.<br>
        • Do not use filters or excessive digital zoom.<br><br>
        <b>Images that fail quality or calibration checks will be rejected and must be retaken.</b><br>
        <span class="muted">The system calculates the ABCDE Diameter (D) from the image and the detected physical calibration reference; the optional size above is not used for D.</span>
        </div>
        """, unsafe_allow_html=True)
    with right:
        if uploaded:
            st.image(Image.open(uploaded).convert("RGB"), caption=uploaded.name)
        else:
            st.info("Image preview")

    if st.button(submit_label, type="primary"):
        if not full_name.strip() or not explanation.strip() or uploaded is None:
            st.error("Name, notes, and image are required.")
            st.stop()

        with st.spinner("Running analysis..."):
            try:
                files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")}
                data = {
                    "patient_name": full_name.strip(),
                    "case_label": full_name.strip(),
                }
                resp = requests.post(API_URL, files=files, data=data, timeout=90)
            except requests.exceptions.RequestException:
                st.error("❌ Connection Error: Could not reach the CEFM backend service.")
                st.info("Please verify that the FastAPI backend server is running: `uvicorn app.main:app --host 0.0.0.0 --port 8000`.")
                st.stop()
            except Exception:
                st.error("❌ Request Error: An unexpected error occurred while sending the image for analysis.")
                st.stop()

            if resp.status_code != 200:
                try:
                    resp_json = resp.json()
                except Exception:
                    resp_json = None

                err = parse_api_error(resp.status_code, resp_json, resp.text)
                st.error(f"❌ {err['title']}: {err['message']}")
                if err.get("guidance"):
                    st.info(f"💡 {err['guidance']}")
                st.stop()

            try:
                result = resp.json()
            except Exception:
                st.error("❌ Response Parsing Error: Backend returned an unparseable response.")
                st.stop()

        patient = {
            "full_name": full_name.strip(),
            "age": age,
            "gender": gender,
            "phone": phone.strip(),
            "visit_date": str(visit_date),
        }
        lesion = {
            "area": lesion_area,
            "size": lesion_size.strip(),
            "duration": duration.strip(),
            "symptoms": symptoms,
            "explanation": explanation.strip(),
        }

        persistence_status, storage_warning = classify_persistence_status(result)

        if should_save_to_history(persistence_status):
            # Confirmed persistence: status='success' and persisted=True
            save_case(save_user_id, patient, lesion, uploaded.name, result)
            st.session_state.unsaved_case = None
            st.session_state.unsaved_storage_warning = None
            st.success(success_msg)
            temp_case = build_temp_case(patient, lesion, uploaded.name, result)
            render_report(temp_case)
        else:
            # Partial success or ambiguous/unconfirmed persistence:
            # Fail safe: NEVER write to SQLite history!
            temp_case = build_temp_case(patient, lesion, uploaded.name, result)
            st.session_state.unsaved_case = temp_case
            st.session_state.unsaved_storage_warning = storage_warning

            st.warning("⚠️ " + (storage_warning or "Analysis completed successfully, but could not be saved to backend database history."))
            st.info(
                "ℹ️ **Session Retention Active**: This clinical analysis is temporarily retained in your browser session "
                "and will remain accessible as you navigate tabs. "
                "**Warning:** This analysis will be permanently lost if you refresh the browser tab or close this session. "
                "Please download a copy of the clinical report below."
            )

            col_dl1, col_dl2 = st.columns([1, 1])
            with col_dl1:
                st.download_button(
                    "📥 Download Clinical Report (.md)",
                    data=generate_report_export(temp_case),
                    file_name=f"CEFM_Report_{temp_case['patient_name']}_{datetime.now():%Y%m%d_%H%M}.md",
                    mime="text/markdown",
                    key="newcase_unsaved_dl_md"
                )
            with col_dl2:
                st.download_button(
                    "📥 Download Assessment Data (JSON)",
                    data=json.dumps(result, indent=2),
                    file_name=f"CEFM_Data_{temp_case['patient_name']}_{datetime.now():%Y%m%d_%H%M}.json",
                    mime="application/json",
                    key="newcase_unsaved_dl_json"
                )

            render_report(temp_case)

# =========================================================
# REPORT RENDER
# =========================================================
def render_report(case):
    result = json.loads(case["result_json"])
    symptoms = json.loads(case.get("symptoms") or "[]")
    clf = result.get("classification", {})
    feat = result.get("features") or result.get("abcde", {})

# Support both flattened and nested ABCDE structures.
    color = feat.get("color", {}) or {}

    rep = result.get("report", {})
    expl = result.get("explanation") or rep.get("explanation", {})
    dia = feat.get("diameter", {}) or {}
    evolution = result.get("evolution") or {}

    pred = clf.get("predicted_class", clf.get("prediction", "—"))
    conf = float(clf.get("melanoma_probability", clf.get("confidence", 0)) or 0)
    risk = rep.get("overall_risk", clf.get("risk_level", "—"))
    risk_kind = "red" if "high" in str(risk).lower() else ("amber" if "medium" in str(risk).lower() or "moderate" in str(risk).lower() else "green")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="card"><div class="card-label">Patient</div>
        <div class="card-value">{case['patient_name']}</div>
        <div class="card-sub">{case['age']} yrs · {case['gender']}</div></div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="card"><div class="card-label">Lesion area</div>
        <div class="card-value">{case['lesion_area']}</div>
        <div class="card-sub">Size: {case['lesion_size'] or '—'}</div></div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="card"><div class="card-label">Duration</div>
        <div class="card-value">{case['duration'] or '—'}</div>
        <div class="card-sub">Visit: {case['visit_date']}</div></div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="card"><div class="card-label">AI assessment</div>
        <div class="card-value">{pred}</div>
        <div class="card-sub">Confidence {conf*100:.1f}%</div></div>
        """, unsafe_allow_html=True)

    st.markdown("**Clinical explanation**")
    st.markdown(f'<div class="box">{case["explanation"]}</div>', unsafe_allow_html=True)
    if symptoms:
        st.caption("Symptoms: " + ", ".join(symptoms))

    st.markdown(f"<div style='margin:0.8rem 0;'>Overall risk: {pill(risk, risk_kind)}</div>", unsafe_allow_html=True)

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.markdown(f"""
        <div class="card"><div class="card-label">Asymmetry</div>
        <div class="card-value">{float(feat.get('asymmetry',0) or 0):.2f}</div>
        <div class="card-sub">A score · 0–1</div></div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown(f"""
        <div class="card"><div class="card-label">Border</div>
        <div class="card-value">{float(feat.get('border_irregularity',0) or 0):.2f}</div>
        <div class="card-sub">B irregularity · 0–1</div></div>
        """, unsafe_allow_html=True)
    with a3:
        st.markdown(f"""
        <div class="card"><div class="card-label">Color variation</div>
        <div class="card-value">{float(feat.get('color_index', color.get('index', 0)) or 0):.2f}</div>
        <div class="card-sub">C index · 0–1</div></div>
        """, unsafe_allow_html=True)
    with a4:
        max_mm = feat.get("diameter_mm")

        if max_mm is not None:
            d_sub = f"{float(max_mm):.2f} mm"
            d_note = "⚠️ exceeds 6 mm" if float(max_mm) > 6 else "D max diameter"
        else:
            max_px = feat.get("diameter_pixels")
            d_sub = f"{float(max_px):.0f} px" if max_px is not None else "Unavailable"
            d_note = "Physical diameter unavailable — calibration required"

        st.markdown(f"""
        <div class="card"><div class="card-label">Diameter</div>
        <div class="card-value">{d_sub}</div>
        <div class="card-sub">{d_note}</div></div>
        """, unsafe_allow_html=True)

    if evolution.get("available"):
        st.markdown("**Evolution (E)**")
        change_text = "Change detected" if evolution.get("change_detected") else "No significant change detected"
        st.markdown(f'''
        <div class="box">
        <b>{change_text}</b><br>
        {evolution.get('description', '')}
        </div>
        ''', unsafe_allow_html=True)
        if evolution.get("changes"):
            st.json(evolution.get("changes", {}))
    else:
        st.markdown("**Evolution (E)**")
        st.markdown(f'''
        <div class="box">
        {evolution.get('reason', 'No previous image is available for longitudinal comparison.')}
        </div>
        ''', unsafe_allow_html=True)

    if expl.get("summary"):
        st.markdown("**AI explanation**")
        st.markdown(f'<div class="box">{expl["summary"]}</div>', unsafe_allow_html=True)

    st.markdown("**Structured report**")
    report_text = rep.get("full_report_text")
    if report_text:
        st.markdown(f'<div class="box box-accent">{report_text.replace(chr(10), "<br><br>")}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="box box-accent">This report was created by an older version. '
            'Please run a new analysis to receive the plain-language report.</div>',
            unsafe_allow_html=True,
        )
    st.info(rep.get("recommendation", ""))

    with st.expander("Technical details"):
        st.json({
            "classification": clf,
            "abcde": feat,
            "calibration": result.get("calibration", {}),
            "validation": rep.get("validation", {}),
            "image_name": case.get("image_name")
        })

    st.markdown("---")
    c_rep1, c_rep2 = st.columns([1, 1])
    with c_rep1:
        st.download_button(
            label="📥 Download Clinical Report (.md)",
            data=generate_report_export(case),
            file_name=f"CEFM_Report_{case.get('patient_name', 'Patient')}_{datetime.now():%Y%m%d_%H%M}.md",
            mime="text/markdown",
            key=f"dl_rep_{case.get('id', 'temp')}_{abs(hash(str(case.get('result_json'))[:50]))}"
        )
    with c_rep2:
        st.download_button(
            label="📥 Download Assessment Data (JSON)",
            data=json.dumps(result, indent=2),
            file_name=f"CEFM_Data_{case.get('patient_name', 'Patient')}_{datetime.now():%Y%m%d_%H%M}.json",
            mime="application/json",
            key=f"dl_json_{case.get('id', 'temp')}_{abs(hash(str(case.get('result_json'))[:50]))}"
        )

# =========================================================
# APP ROUTING
# =========================================================
def main():
    if "user" not in st.session_state:
        st.session_state.user = None
    if "unsaved_case" not in st.session_state:
        st.session_state.unsaved_case = None
    if "unsaved_storage_warning" not in st.session_state:
        st.session_state.unsaved_storage_warning = None

    if st.session_state.user is None:
        auth_view()
        st.stop()

    user = st.session_state.user
    role = user["role"]

    with st.sidebar:
        st.markdown("### CEFM")
        st.caption(f"{user['full_name']} · {role.capitalize()}")
        st.markdown("---")
        if role == "patient":
            menu = st.radio("Navigation", ["New Case", "My History", "About"], label_visibility="collapsed")
        else:
            menu = st.radio(
                "Navigation",
                ["New Case for Patient", "All Cases", "About"],
                label_visibility="collapsed"
            )
        st.markdown("---")
        if api_online():
            st.success("Backend online")
        else:
            st.error("Backend offline")
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.unsaved_case = None
            st.rerun()

    # ---------------- PATIENT: NEW CASE ----------------
    if role == "patient" and menu == "New Case":
        st.markdown("""
        <div class="hero">
          <h1>New Case Assessment</h1>
          <p>Enter your details, lesion information, and upload a dermoscopic image.</p>
        </div>
        """, unsafe_allow_html=True)

        case_form_and_analyze(
            submit_label="Submit & analyze",
            notes_label="Explanation / clinical notes *",
            success_msg="Case analyzed and saved. Open **My History**.",
            save_user_id=user["id"]
        )

    # ---------------- PATIENT HISTORY ----------------
    elif role == "patient" and menu == "My History":
        st.markdown("""
        <div class="hero">
          <h1>My History</h1>
          <p>Previously submitted cases and reports.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("unsaved_case"):
            unsaved = st.session_state.unsaved_case
            st.warning(
                "⚠️ **Unsaved Analysis Active in Current Session**: You have an unpersisted clinical assessment from this session. "
                "Because database persistence failed or was unconfirmed, it is NOT stored in the database history below."
            )
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                st.download_button(
                    "📥 Download Unsaved Report (.md)",
                    data=generate_report_export(unsaved),
                    file_name=f"CEFM_Unsaved_{unsaved['patient_name']}_{datetime.now():%Y%m%d_%H%M}.md",
                    mime="text/markdown",
                    key="patient_hist_unsaved_dl_md"
                )
            with c2:
                if st.session_state.get("view_unsaved_patient_history"):
                    if st.button("Hide Unsaved Analysis", key="patient_hide_unsaved_hist"):
                        st.session_state.view_unsaved_patient_history = False
                        st.rerun()
                else:
                    if st.button("👁️ View Unsaved Analysis", key="patient_view_unsaved_hist"):
                        st.session_state.view_unsaved_patient_history = True
                        st.rerun()
            with c3:
                if st.button("Discard Unsaved Case", key="patient_discard_unsaved_hist"):
                    st.session_state.unsaved_case = None
                    st.session_state.view_unsaved_patient_history = False
                    st.rerun()

            if st.session_state.get("view_unsaved_patient_history"):
                st.markdown("---")
                st.markdown("### 📋 Unsaved Assessment (Current Session)")
                render_report(unsaved)
                st.markdown("---")

        rows = cases_for_patient(user["id"])
        if not rows:
            st.info("No cases yet. Create one in New Case.")
        else:
            labels = {
                f"#{r['id']}  ·  {r['lesion_area']}  ·  {r['created_at'][:16]}  ·  {json.loads(r['result_json']).get('classification',{}).get('prediction','?')}": r["id"]
                for r in rows
            }
            choice = st.selectbox("Select case", list(labels.keys()))
            render_report(get_case(labels[choice]))

    # ---------------- DOCTOR: NEW CASE FOR PATIENT ----------------
    elif role == "doctor" and menu == "New Case for Patient":
        st.markdown("""
        <div class="hero">
          <h1>New Case for Patient</h1>
          <p>Upload and analyze a lesion image on behalf of your patient.</p>
        </div>
        """, unsafe_allow_html=True)

        case_form_and_analyze(
            submit_label="Analyze patient case",
            notes_label="Doctor clinical notes *",
            success_msg="Patient case analyzed and saved. Open **All Cases**.",
            save_user_id=user["id"]
        )

    # ---- DOCTOR: ALL CASES ----
    elif role == "doctor" and menu == "All Cases":
        st.markdown("""
        <div class="hero">
          <h1>All Cases</h1>
          <p>Every analyzed case across all patients.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("unsaved_case"):
            unsaved = st.session_state.unsaved_case
            st.warning(
                "⚠️ **Unsaved Analysis Active in Current Session**: You have an unpersisted patient assessment from this session. "
                "Because database persistence failed or was unconfirmed, it is NOT stored in the database history below."
            )
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                st.download_button(
                    "📥 Download Unsaved Report (.md)",
                    data=generate_report_export(unsaved),
                    file_name=f"CEFM_Unsaved_{unsaved['patient_name']}_{datetime.now():%Y%m%d_%H%M}.md",
                    mime="text/markdown",
                    key="doctor_hist_unsaved_dl_md"
                )
            with c2:
                if st.session_state.get("view_unsaved_doctor_history"):
                    if st.button("Hide Unsaved Analysis", key="doctor_hide_unsaved_hist"):
                        st.session_state.view_unsaved_doctor_history = False
                        st.rerun()
                else:
                    if st.button("👁️ View Unsaved Analysis", key="doctor_view_unsaved_hist"):
                        st.session_state.view_unsaved_doctor_history = True
                        st.rerun()
            with c3:
                if st.button("Discard Unsaved Case", key="doctor_discard_unsaved_hist"):
                    st.session_state.unsaved_case = None
                    st.session_state.view_unsaved_doctor_history = False
                    st.rerun()

            if st.session_state.get("view_unsaved_doctor_history"):
                st.markdown("---")
                st.markdown("### 📋 Unsaved Assessment (Current Session)")
                render_report(unsaved)
                st.markdown("---")

        rows = all_cases()
        if not rows:
            st.info("No cases yet. Create one in **New Case for Patient**.")
        else:
            labels = {
                f"#{r['id']}  ·  {r['patient_name']}  ·  {r['lesion_area']}  ·  {r['created_at'][:16]}": r["id"]
                for r in rows
            }
            choice = st.selectbox("Select case", list(labels.keys()))
            render_report(get_case(labels[choice]))

    # ---------------- ABOUT ----------------
    elif menu == "About":
        st.markdown("""
        <div class="hero">
          <h1>About CEFM</h1>
          <p>Explainable lesion assessment with patient history and doctor review.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        **Pipeline**: image validation → image type detection → lesion segmentation
        → ABCDE features (Asymmetry, Border, Color, Diameter, Evolution)
        → classification → structured report.

        - **A**symmetry — symmetry of the lesion about its axes
        - **B**order — irregularity of the lesion contour
        - **C**olor — colour variation inside the lesion
        - **D**iameter — maximum lesion diameter (6 mm warning threshold)
        - **E**volution — change versus a previous analysis of the same case

        ⚠️ For research and clinical decision support only — not a diagnosis.
        """)

    # ---- FALLBACK ----
    else:
        st.info("Select an option from the sidebar.")


if __name__ == "__main__":
    main()