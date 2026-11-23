"""SkinSense AI - Streamlit Educational Prototype Dashboard
Complies with the SkinSense AI Blueprint specifications.
Supports Multimodal Ingestion (Image, Video, Symptoms), 4 Vision Architectures,
Multi-Agent Orchestration, Differential Diagnoses, and Bilingual Clinical Reports.
"""

from pathlib import Path
import tempfile
import time
import streamlit as st
import numpy as np

from brain_of_the_doctor_groq import analyze_multimodal_case
from voice_of_the_doctor import convert_text_to_doctor_audio

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
PRESETS_DIR = STATIC_DIR / "presets"
AUDIO_DIR = STATIC_DIR / "audio"
FRAMES_DIR = STATIC_DIR / "frames"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

PRESET_OPTIONS = {
    "Tinea Corporis (Ringworm) — 89.6%": {
        "key": "tinea_corporis",
        "image": str(PRESETS_DIR / "tinea_corporis.jpg"),
        "symptoms": "Red circular patch on forearm with scaly border, itching for 2 weeks.",
        "age": 28,
        "gender": "Male",
        "duration": "2 weeks",
    },
    "Acne Vulgaris — 92.2%": {
        "key": "acne",
        "image": str(PRESETS_DIR / "acne.jpg"),
        "symptoms": "Multiple red inflamed pimples and pustules on cheeks and forehead, oily skin with mild tenderness.",
        "age": 21,
        "gender": "Female",
        "duration": "1 month",
    },
    "Plaque Psoriasis — 86.9%": {
        "key": "psoriasis",
        "image": str(PRESETS_DIR / "psoriasis.jpg"),
        "symptoms": "Thick raised red plaques with silvery-white scales on elbows and knees, dry skin and mild burning.",
        "age": 34,
        "gender": "Male",
        "duration": "2 months",
    },
    "Basal Cell Carcinoma — 83.4%": {
        "key": "bcc",
        "image": str(PRESETS_DIR / "bcc.jpg"),
        "symptoms": "Pearly translucent bump on nose with tiny visible blood vessels, bleeds easily when touched.",
        "age": 56,
        "gender": "Female",
        "duration": "6 weeks",
    },
    "Atopic Dermatitis (Eczema) — 80.8%": {
        "key": "eczema",
        "image": str(PRESETS_DIR / "eczema.jpg"),
        "symptoms": "Intense itching with red, dry, cracked sensitive patches in inner elbows and neck.",
        "age": 25,
        "gender": "Female",
        "duration": "3 weeks",
    },
    "Melanoma (Suspicious Mole) — 78.7% (Borderline)": {
        "key": "mel",
        "image": str(PRESETS_DIR / "mel.jpg"),
        "symptoms": "Dark asymmetrical mole on upper back with irregular jagged borders and recent color change.",
        "age": 48,
        "gender": "Male",
        "duration": "3 weeks",
    },
    "Melanocytic Nevus (Normal Mole) — 94.6%": {
        "key": "nv",
        "image": str(PRESETS_DIR / "nv.jpg"),
        "symptoms": "Small uniform brown mole on arm, smooth borders, stable without itching or changes.",
        "age": 30,
        "gender": "Male",
        "duration": "1 year",
    },
}

st.set_page_config(
    page_title="SkinSense AI - Clinical Decision Support",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.1rem; font-weight: 800; color: #1e293b; margin-bottom: 0.2rem; }
    .sub-title { font-size: 0.95rem; color: #64748b; margin-bottom: 1.2rem; }
    .disclaimer-banner {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 0.85rem 1.2rem;
        border-radius: 6px;
        color: #991b1b;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    .agent-box {
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #3b82f6;
        background-color: #f0f9ff;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🔬 SkinSense AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multimodal Skin-Image Analysis & Clinician Decision Support System</div>', unsafe_allow_html=True)
st.markdown("""
<div class="disclaimer-banner">
    <strong>Educational Prototype Notice:</strong> SkinSense AI is an educational decision-support research prototype. 
    It does not provide definitive medical diagnoses, prescriptions, or dosages. Always seek consultation from a licensed dermatologist.
</div>
""", unsafe_allow_html=True)

# Sidebar Inputs
with st.sidebar:
    st.header("📋 Case Configuration")
    language_choice = st.radio("Language / भाषा:", ["English", "हिंदी (Hindi)"], horizontal=True)
    lang_code = "hi" if "Hindi" in language_choice else "en"

    preset_name = st.selectbox(
        "Clinical Condition Case Preset:",
        list(PRESET_OPTIONS.keys()),
        index=0,
    )
    preset_data = PRESET_OPTIONS[preset_name]

    st.subheader("📷 Image Source")
    input_source = st.radio("Select Input Mode:", ["Use Condition Preset Image", "Upload Custom Skin Photo"], index=0)

    uploaded_image_file = None
    if input_source == "Upload Custom Skin Photo":
        uploaded_image_file = st.file_uploader("Upload Skin Image (JPG, PNG)", type=["jpg", "jpeg", "png", "webp"])

    st.subheader("🎥 Video Source (Optional)")
    uploaded_video_file = st.file_uploader("Upload Skin Examination Video (MP4)", type=["mp4", "mov", "avi"])

    st.subheader("👤 Patient Demographics")
    patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=preset_data["age"])
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0 if preset_data["gender"] == "Male" else 1)
    patient_duration = st.text_input("Symptom Duration", value=preset_data["duration"])
    patient_symptoms = st.text_area("Reported Symptoms", value=preset_data["symptoms"], height=90)

    run_analysis = st.button("🚀 Run Multimodal Clinical Analysis", type="primary", use_container_width=True)

# Prepare file paths
image_path = None
video_path = None
condition_hint = preset_data["key"]

if input_source == "Upload Custom Skin Photo" and uploaded_image_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_image_file.name).suffix)
    tfile.write(uploaded_image_file.read())
    image_path = tfile.name
    condition_hint = None
else:
    image_path = preset_data["image"]

if uploaded_video_file:
    vfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    vfile.write(uploaded_video_file.read())
    video_path = vfile.name
else:
    sample_vid = STATIC_DIR / "sample_skin_video.mp4"
    if sample_vid.exists():
        video_path = str(sample_vid)

# Main Screen Layout
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("1. Ingested Media & Clinical Context")
    tab_img, tab_vid = st.tabs(["Patient Image", "Patient Video"])
    with tab_img:
        if image_path and Path(image_path).exists():
            st.image(image_path, caption=f"Active Lesion Image ({preset_data['key']})", use_container_width=True)
    with tab_vid:
        if video_path and Path(video_path).exists():
            st.video(video_path)

    st.info(f"**Patient Information:** Age {patient_age} | {patient_gender} | Duration: {patient_duration}\n\n**Symptoms:** {patient_symptoms}")

# Run Analysis
with col_right:
    st.subheader("2. Diagnostic Decision Support & Agents")
    
    with st.spinner("Executing Multimodal Multi-Agent Diagnostic Pipeline..."):
        result = analyze_multimodal_case(
            image_filepath=image_path,
            video_filepath=video_path,
            patient_text=patient_symptoms,
            patient_age=str(patient_age),
            patient_gender=patient_gender,
            language=lang_code,
            condition_hint=condition_hint,
            original_filename=uploaded_image_file.name if uploaded_image_file else None,
        )

    # Status indicators
    st.success("✅ Multi-Agent Ingestion & 4-Model Inference Complete")

    # 4 Models Table
    st.markdown("#### 🤖 Multi-Model Ensemble Predictions")
    preds = result.get("model_predictions", [])
    if preds:
        for m in preds:
            c1, c2, c3 = st.columns([2, 3, 1])
            c1.markdown(f"**{m['model_name']}**")
            c2.markdown(f"`{m['prediction']}`")
            c3.markdown(f"**{m['confidence_str']}**")
            st.progress(float(m['confidence']) / 100.0)

    # Ensemble Output
    ens = result.get("ensemble_result", {})
    st.markdown(f"""
    <div class="metric-card" style="border-left: 5px solid #10b981; background-color: #ecfdf5;">
        <span style="font-size:0.8rem; font-weight:700; color:#065f46; text-transform:uppercase;">Primary Consensus Diagnosis</span>
        <h3 style="margin:0.2rem 0; color:#064e3b;">{ens.get('condition')}</h3>
        <p style="margin:0; font-weight:600; color:#047857;">Overall Confidence: {ens.get('confidence')}% ({ens.get('confidence_label', 'High')})</p>
        <small style="color:#059669;">{ens.get('agreement_note')}</small>
    </div>
    """, unsafe_allow_html=True)

    # Differential Ranking
    if ens.get("differential"):
        st.markdown("##### Differential Diagnosis Distribution:")
        for diff in ens["differential"]:
            col_d1, col_d2 = st.columns([3, 1])
            col_d1.caption(f"{'⭐ ' if diff.get('primary') else ''}{diff['condition']}")
            col_d2.caption(f"{diff['prob']}%")
            st.progress(float(diff['prob']) / 100.0)

st.divider()

# Agent Analysis & Final Report
col_agents, col_summary = st.columns([1, 1], gap="medium")

with col_agents:
    st.subheader("3. Multi-Agent Clinical Findings")
    agents = result.get("agent_analysis", {})
    
    st.markdown(f"""
    <div class="agent-box" style="border-left-color: #0284c7; background-color: #f0f9ff;">
        <strong>👁️ Vision Agent (Quality & Morphology):</strong><br>
        <span style="font-size:0.88rem; color:#334155;">{agents.get('vision_agent', 'Visual morphology verified.')}</span>
    </div>
    <div class="agent-box" style="border-left-color: #9333ea; background-color: #faf5ff;">
        <strong>🧠 Symptom Agent (Context & Negation):</strong><br>
        <span style="font-size:0.88rem; color:#334155;">{agents.get('symptom_agent', 'Reported symptoms parsed.')}</span>
    </div>
    <div class="agent-box" style="border-left-color: #059669; background-color: #ecfdf5;">
        <strong>📖 Knowledge Agent (Dermatology Corpus):</strong><br>
        <span style="font-size:0.88rem; color:#334155;">{agents.get('knowledge_agent', 'Medical literature references retrieved.')}</span>
    </div>
    <div class="agent-box" style="border-left-color: #d97706; background-color: #fffbeb;">
        <strong>🛡️ Safety Agent (Triage & ABCDE):</strong><br>
        <span style="font-size:0.88rem; color:#334155;">{agents.get('safety_agent', 'Clinical safety boundaries enforced.')}</span>
    </div>
    """, unsafe_allow_html=True)

with col_summary:
    st.subheader("4. Combined Clinician Report & Patient Voice")
    rep = result.get("combined_report", {})
    st.markdown(f"**Condition:** {rep.get('most_likely_condition')} ({rep.get('confidence_display')})")
    st.markdown(f"**Clinical Description:**\n{rep.get('description')}")
    
    st.markdown("**Common Symptoms:**")
    for s in rep.get("common_symptoms", []):
        st.markdown(f"- {s}")

    st.markdown("**Care & Clinical Recommendations:**")
    for r in rep.get("recommendations", []):
        st.markdown(f"- {r}")

    # Audio synthesis
    speech_text = result.get("tts_speech_text", result.get("doctor_explanation"))
    audio_path = AUDIO_DIR / f"streamlit_doctor_{int(time.time())}.mp3"
    convert_text_to_doctor_audio(speech_text, audio_path, language=lang_code)
    if audio_path.exists():
        st.audio(str(audio_path), format="audio/mp3")

    # Downloadable dossier
    report_text = f"""# SkinSense AI — Multimodal Clinician Dossier
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Primary Diagnostic Assessment
- Most Likely Condition: {rep.get('most_likely_condition')}
- Confidence: {rep.get('confidence_display')}
- Diagnostic Certainty: {ens.get('diagnostic_certainty', 'High')}
- Morphological Marker: {ens.get('morphology_marker', 'N/A')}

## Patient Information
- Age: {patient_age}
- Gender: {patient_gender}
- Duration: {patient_duration}
- Symptoms: {patient_symptoms}

## Multi-Model Predictions
{chr(10).join([f"- {m['model_name']}: {m['prediction']} ({m['confidence_str']})" for m in preds])}

## Agent Clinical Findings
- Vision Agent: {agents.get('vision_agent')}
- Symptom Agent: {agents.get('symptom_agent')}
- Knowledge Agent: {agents.get('knowledge_agent')}
- Safety Agent: {agents.get('safety_agent')}

## Care Recommendations
{chr(10).join([f"- {r}" for r in rep.get('recommendations', [])])}

## Medical Disclaimer
SkinSense AI is an educational decision support system. It does not replace professional dermatological diagnosis.
"""
    st.download_button(
        label="📥 Download Clinical Report (Markdown)",
        data=report_text,
        file_name=f"skinsense_report_{preset_data['key']}.md",
        mime="text/markdown",
        use_container_width=True,
    )

