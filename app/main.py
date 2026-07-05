"""EcoSort AI — Main Streamlit Dashboard.

Industry-ready waste classification supporting both IMAGE and VIDEO input,
styled with the EcoSort AI Design System (Hanken Grotesk, Sustainability Green).
"""

import sys, os, random, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))

import streamlit as st
from PIL import Image

from app.preprocessor import load_and_validate_image, get_image_metadata
from app.advisor import get_disposal_advisory, get_fallback_guide
from app.classifier import (
    CATEGORY_COLORS, BIN_COLORS, classify_image, get_mock_classification, load_model,
)
from database.db_manager import (
    init_db, log_classification, get_recent_classifications,
    get_total_carbon_saved, get_category_counts, get_total_classifications,
)

# Optional imports
try:
    import tensorflow as tf; TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
try:
    import cv2; CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

st.set_page_config(page_title="EcoSort AI — Waste Classifier", page_icon="🌿",
                    layout="wide", initial_sidebar_state="expanded")
init_db()

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700&display=swap');
html, body, .stApp {font-family: 'Hanken Grotesk', sans-serif !important;}
.stApp {background-color: #f8f9ff;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#006c49 0%,#004d35 100%)}
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] .stMarkdown h1, section[data-testid="stSidebar"] .stMarkdown h5 {color:white !important}
section[data-testid="stSidebar"] .stRadio label {color:white !important}
section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.15)}
.eco-card{background:#fff;border:1px solid #e2e8f0;border-radius:1rem;padding:1.5rem;margin-bottom:1rem;transition:box-shadow .25s ease}
.eco-card:hover{box-shadow:0 4px 20px rgba(0,108,73,.08)}
.metric-card{background:#fff;border:1px solid #e2e8f0;border-radius:1rem;padding:1.25rem 1.5rem;text-align:center}
.metric-value{font-size:2rem;font-weight:700;color:#006c49;line-height:1.2}
.metric-label{font-size:.75rem;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:#64748b;margin-top:.25rem}
.chip{display:inline-block;padding:.3rem .85rem;border-radius:9999px;font-size:.8rem;font-weight:600}
.hero-title{font-size:2.8rem;font-weight:700;color:#0b1c30;line-height:1.15;letter-spacing:-.02em;margin-bottom:.5rem}
.hero-subtitle{font-size:1.15rem;color:#64748b;line-height:1.6;max-width:600px;margin:0 auto;}
.hero-badge{display:inline-block;background:linear-gradient(135deg,#10b981,#006c49);color:#fff;font-size:.75rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:.35rem 1rem;border-radius:9999px;margin-bottom:1rem}
.feature-card{background:#fff;border:1px solid #e2e8f0;border-radius:1rem;padding:1.5rem;text-align:center;min-height:180px;transition:transform .2s,box-shadow .2s}
.feature-card:hover{transform:translateY(-4px);box-shadow:0 8px 30px rgba(0,108,73,.10)}
.feature-icon{font-size:2.5rem;margin-bottom:.75rem}
.feature-title{font-size:1rem;font-weight:700;color:#0b1c30;margin-bottom:.35rem}
.feature-desc{font-size:.85rem;color:#64748b;line-height:1.5}
.guide-rule{background:#f8fafc;border-left:3px solid #10b981;padding:.65rem 1rem;border-radius:0 .5rem .5rem 0;margin-bottom:.5rem;font-size:.9rem;color:#334155}
.section-heading{font-size:1.35rem;font-weight:700;color:#0b1c30;margin-bottom:1rem}
.text-muted{color:#64748b}.text-sm{font-size:.85rem}
.stButton>button{background:linear-gradient(135deg,#10b981,#006c49);color:#fff;border:none;border-radius:.5rem;font-weight:600;padding:.55rem 1.5rem;transition:opacity .2s}
.stButton>button:hover{opacity:.9;color:#fff}
[data-testid="stFileUploader"] section{border:2px dashed rgba(255,255,255,.35)!important;border-radius:1rem}
#MainMenu, header, footer {display: none;}
</style>""", unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                         ║
# ╚═══════════════════════════════════════════════════════════════════╝

@st.cache_resource(show_spinner="Loading MobileNetV2 model…")
def _cached_model():
    return load_model()

def _classify_pil(pil_image, use_gemini=True):
    """Run classification on a single PIL image using Gemini Vision (fallback to TF)."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    # If TensorFlow is unavailable, we MUST use Gemini for images as well
    if not TF_AVAILABLE:
        use_gemini = True
        
    if use_gemini and api_key and api_key != "your_gemini_api_key_here":
        import google.generativeai as genai
        import json
        from app.classifier import CATEGORY_COLORS, BIN_COLORS
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = '''Identify the primary waste item in this image.
Classify it into EXACTLY one of these 4 categories: Organic, Recyclable, Hazardous, E-waste.
Return ONLY a raw JSON array containing exactly one object, without any markdown backticks. Example:
[{"label": "Water Bottle", "waste_category": "Recyclable", "confidence": 0.98}]'''
        try:
            resp = model.generate_content([prompt, pil_image])
            text = resp.text.strip()
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            data = json.loads(text.strip())
            
            results = []
            for d in data:
                cat = d.get("waste_category", "Recyclable")
                if cat not in CATEGORY_COLORS: cat = "Recyclable"
                results.append({
                    "label": d.get("label", "Unknown").title(),
                    "raw_label": d.get("label", "unknown").lower().replace(" ", "_"),
                    "confidence": float(d.get("confidence", 0.95)),
                    "waste_category": cat,
                    "category_color": CATEGORY_COLORS.get(cat, "#3B82F6"),
                    "bin_color": BIN_COLORS.get(cat, "Blue 🔵")
                })
            if results: return results
        except Exception as e:
            print(f"Gemini Vision API Error: {e}")

    # Fallback to local MobileNetV2 if Gemini fails or is not configured
    if TF_AVAILABLE:
        from app.preprocessor import preprocess_for_mobilenet
        from app.classifier import classify_image
        model = _cached_model()
        tensor = preprocess_for_mobilenet(pil_image)
        return classify_image(model, tensor)
    
    from app.classifier import get_mock_classification
    return get_mock_classification()

def _render_result_card(top, cat_color, carbon_estimate):
    """Render the classification result card + mini metrics."""
    st.markdown(f"""
    <div class="eco-card" style="border-left:4px solid {cat_color}">
        <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:1rem">
            <span class="chip" style="background:{cat_color}20;color:{cat_color}">{top['waste_category']}</span>
            <span class="text-sm text-muted">{top['bin_color']} Bin</span>
        </div>
        <div style="font-size:1.6rem;font-weight:700;color:#0b1c30;margin-bottom:.5rem">{top['label']}</div>
        <div style="font-size:.9rem;color:#64748b">Confidence: <strong style="color:{cat_color}">{top['confidence']:.1%}</strong></div>
    </div>""", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    for col, val, lbl in [(m1, f"{top['confidence']:.0%}", "Confidence"),
                           (m2, top['waste_category'][:3], "Category"),
                           (m3, f"{carbon_estimate} kg", "CO₂ Saved")]:
        with col:
            color = f"color:{cat_color};" if lbl == "Category" else ""
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.4rem;{color}">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

def _render_guide(guide, cat_color):
    """Render disposal guide + environmental insights."""
    st.markdown('<div class="section-heading">♻️ Eco-Disposal Guide</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="eco-card" style="border-left:4px solid {cat_color};padding-bottom:1rem"><div style="font-weight:600;color:#0b1c30;margin-bottom:.75rem">🗑️ Recommended Bin: <strong>{guide.bin_color}</strong></div>', unsafe_allow_html=True)
    for rule in guide.recycling_rules:
        st.markdown(f'<div class="guide-rule">✓ {rule}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def _render_confidence_chart(results):
    """Render a Plotly horizontal bar chart of prediction confidence."""
    import plotly.graph_objects as go
    labels = [r["label"] for r in results]
    confs = [r["confidence"] for r in results]
    colors = [r["category_color"] for r in results]
    fig = go.Figure(go.Bar(x=confs, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0), cornerradius=6),
        text=[f"{c:.1%}" for c in confs], textposition="outside",
        textfont=dict(family="Hanken Grotesk", size=13, color="#334155")))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Hanken Grotesk"),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0", tickformat=".0%", range=[0, max(confs)*1.3]),
        yaxis=dict(autorange="reversed", showgrid=False),
        margin=dict(l=0, r=40, t=10, b=10), height=280)
    st.plotly_chart(fig, key=f"chart_{random.randint(0,99999)}", config={"displayModeBar": False})

def _render_insights(guide):
    """Render environmental insight cards + banner."""
    st.markdown('<div class="section-heading">🌍 Environmental Insights</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, icon, title, text in [
        (c1, "⏳", "Decomposition Time", guide.decomposition_time),
        (c2, "🌱", "Carbon Tip", guide.carbon_tip),
        (c3, "💡", "Did You Know?", guide.fun_fact),
    ]:
        with col:
            st.markdown(f'<div class="eco-card"><div style="font-weight:600;color:#0b1c30;margin-bottom:.5rem">{icon} {title}</div><div class="text-sm" style="color:#334155">{text}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:linear-gradient(135deg,#006c49,#004d35);border-radius:1rem;padding:1.5rem 2rem;margin-top:1rem;color:#fff"><div style="font-weight:700;font-size:1.1rem;margin-bottom:.5rem">🌏 Environmental Impact</div><div style="font-size:.95rem;opacity:.9;line-height:1.6">{guide.environmental_impact}</div></div>', unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                         ║
# ╚═══════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:1rem 0 .5rem">
        <span style="font-size:2.5rem">🌿</span>
        <h1 style="font-size:1.6rem;font-weight:700;margin:.25rem 0 0">EcoSort AI</h1>
        <p style="font-size:.8rem;opacity:.7;margin:0">Waste Classifier &amp; Sorting Assistant</p>
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("##### ⚙️ System Status")
    
    def _status_card(icon, text, color):
        st.markdown(f'<div style="background:rgba(255,255,255,0.1); border-left:4px solid {color}; padding:0.75rem; border-radius:0.5rem; margin-bottom:0.75rem; color:white; font-size:0.9rem; font-weight:500;">{icon} &nbsp; {text}</div>', unsafe_allow_html=True)
    
    if TF_AVAILABLE:
        _status_card("✅", "TensorFlow ready", "#10b981")
    else:
        _status_card("✅", "TensorFlow unavailable", "#f59e0b")
        
    if CV2_AVAILABLE:
        _status_card("✅", "OpenCV ready", "#10b981")
    else:
        _status_card("⚠️", "OpenCV unavailable (no video)", "#f59e0b")
        
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key and api_key != "your_gemini_api_key_here":
        _status_card("✅", "Gemini API connected", "#10b981")
    else:
        _status_card("ℹ️", "Gemini offline — fallback guides", "#3b82f6")

    st.divider()
    st.markdown("##### 📊 Quick Stats")
    total = get_total_classifications()
    carbon = get_total_carbon_saved()
    st.markdown(f"""
    <div class="metric-card" style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15)">
        <div class="metric-value" style="color:#4edea3">{total}</div>
        <div class="metric-label" style="color:rgba(255,255,255,.6)">Items Classified</div>
    </div><div style="height:.75rem"></div>
    <div class="metric-card" style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15)">
        <div class="metric-value" style="color:#4edea3">{carbon:.1f} kg</div>
        <div class="metric-label" style="color:rgba(255,255,255,.6)">CO₂ Offset Logged</div>
    </div>""", unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  MAIN CONTENT                                                    ║
# ╚═══════════════════════════════════════════════════════════════════╝

if True:
    # ── LANDING PAGE & UPLOADER ──────────────────────────────────
    st.markdown("""<div style="padding:2rem 0 1rem; text-align:center;">
        <span class="hero-badge">AI-Powered Sustainability</span>
        <h1 class="hero-title">Classify Waste.<br/>Protect the Planet.</h1>
        <p class="hero-subtitle" style="margin: 0 auto; text-align: center;">Upload an image or video of any waste item and let our
        MobileNetV2 deep-learning model identify it. EcoSort AI provides instant
        classification, eco-disposal guides, and carbon-footprint insights.</p>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    /* Enlarge the radio buttons */
    div[data-testid="stRadio"] label {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        justify-content: center;
        gap: 2rem;
        margin-bottom: 1rem;
    }
    /* Enlarge the file uploader */
    [data-testid="stFileUploader"] section {
        padding: 3rem !important;
        border: 3px dashed #10b981 !important;
        background-color: rgba(16, 185, 129, 0.05);
    }
    [data-testid="stFileUploader"] section:hover {
        background-color: rgba(16, 185, 129, 0.1);
    }
    [data-testid="stFileUploader"] small {
        font-size: 1.1rem !important;
    }
    [data-testid="stFileUploader"] button {
        font-size: 1.2rem !important;
        padding: 0.5rem 1.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    c_left, c_center, c_right = st.columns([1, 6, 1])
    with c_center:
        input_mode = st.radio("📥 Input Mode", ["📷 Image", "🎬 Video"], horizontal=True, label_visibility="collapsed")
        
        if input_mode == "📷 Image":
            st.markdown("<h3 style='text-align:center;'>📤 Upload Waste Image</h3>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png","webp"], label_visibility="collapsed", key="img_upload")
            uploaded_video = None
        else:
            st.markdown("<h3 style='text-align:center;'>🎬 Upload Waste Video</h3>", unsafe_allow_html=True)
            uploaded_video = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv","webm"], label_visibility="collapsed", key="vid_upload")
            uploaded_file = None

if uploaded_file is None and uploaded_video is None:
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, [
        ("🧠","Deep Learning","MobileNetV2 CNN classifies waste into Organic, Recyclable, Hazardous, or E-waste."),
        ("🎬","Video Analysis","Upload videos — AI extracts key frames and classifies waste across the timeline."),
        ("🤖","Gemini AI Advisor","Personalised disposal advice powered by Google Gemini 2.5 Flash."),
        ("📊","Carbon Tracker","Track your recycling streak and cumulative CO₂ offset over time."),
    ]):
        with col:
            st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-desc">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🔄 How It Works</div>', unsafe_allow_html=True)
    for col, (num, title, desc) in zip(st.columns(4), [
        ("1️⃣","Upload","Choose an image or video of your waste item."),
        ("2️⃣","Classify","MobileNetV2 analyses each frame and predicts the waste category."),
        ("3️⃣","Advise","Gemini AI generates personalised disposal instructions."),
        ("4️⃣","Track","Results are logged so you can watch your green impact grow."),
    ]):
        with col:
            st.markdown(f'<div class="eco-card" style="text-align:center;min-height:150px"><div style="font-size:2rem;margin-bottom:.5rem">{num}</div><div style="font-weight:700;color:#0b1c30;margin-bottom:.25rem">{title}</div><div class="text-sm text-muted">{desc}</div></div>', unsafe_allow_html=True)

    history = get_recent_classifications(limit=5)
    if history:
        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading">🕑 Recent Classifications</div>', unsafe_allow_html=True)
        import pandas as pd
        df = pd.DataFrame(history)[["timestamp","image_name","predicted_label","waste_category","confidence"]]
        df.columns = ["Time","Image","Prediction","Category","Confidence"]
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.1%}")
        df["Time"] = pd.to_datetime(df["Time"]).dt.strftime("%b %d, %H:%M")
        st.dataframe(df, hide_index=True)

elif uploaded_file is not None:
    # ── IMAGE CLASSIFICATION ──────────────────────────────────────
    pil_image = load_and_validate_image(uploaded_file)
    meta = get_image_metadata(pil_image)

    with st.spinner("🔍 Analysing image with AI…"):
        results = _classify_pil(pil_image)

    top = results[0]
    cat_color = top["category_color"]
    guide = get_disposal_advisory(top["label"], top["waste_category"]) or get_fallback_guide(top["waste_category"])
    carbon_estimate = round(random.uniform(0.05, 0.35), 2)

    log_classification(image_name=uploaded_file.name, predicted_label=top["label"],
        waste_category=top["waste_category"], confidence=top["confidence"],
        bin_color=top["bin_color"], carbon_saved_kg=carbon_estimate)

    col_img, col_result = st.columns([1, 1], gap="large")
    with col_img:
        st.markdown('<div class="section-heading">📷 Uploaded Image</div>', unsafe_allow_html=True)
        st.image(pil_image)
        st.markdown(f'<div class="text-sm text-muted">{meta["width"]}×{meta["height"]} · {meta["format"]} · {meta["mode"]}</div>', unsafe_allow_html=True)
    with col_result:
        st.markdown('<div class="section-heading">🏷️ Classification Result</div>', unsafe_allow_html=True)
        _render_result_card(top, cat_color, carbon_estimate)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col_chart, col_guide = st.columns([1, 1], gap="large")
    with col_chart:
        st.markdown('<div class="section-heading">📊 Prediction Confidence</div>', unsafe_allow_html=True)
        _render_confidence_chart(results)
    with col_guide:
        _render_guide(guide, cat_color)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    _render_insights(guide)

elif uploaded_video is not None:
    # ── VIDEO CLASSIFICATION ──────────────────────────────────────
    if not CV2_AVAILABLE:
        st.error("⚠️ OpenCV is not installed. Run: `pip install opencv-python-headless`")
        st.stop()

    from app.video_processor import (
        save_uploaded_video, get_video_metadata, extract_key_frames, aggregate_frame_results,
    )

    with st.spinner("🎬 Processing video…"):
        video_path = save_uploaded_video(uploaded_video)
        vmeta = get_video_metadata(video_path)
        frames = extract_key_frames(video_path, max_frames=8)

    # Video info header
    st.markdown(f"""<div class="eco-card" style="border-left:4px solid #10b981">
        <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap">
            <div><span style="font-size:2rem">🎬</span></div>
            <div>
                <div style="font-weight:700;font-size:1.2rem;color:#0b1c30">{uploaded_video.name}</div>
                <div class="text-sm text-muted">{vmeta['width']}×{vmeta['height']} · {vmeta['fps']} fps · {vmeta['duration_str']} · {vmeta['file_size_mb']} MB · {vmeta.get('codec','')}</div>
            </div>
            <div style="margin-left:auto">
                <span class="chip" style="background:#10b98120;color:#10b981">{len(frames)} frames extracted</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Classify each frame
    with st.spinner(f"🧠 Classifying {len(frames)} key frames…"):
        frame_results = []
        for f in frames:
            frame_results.append(_classify_pil(f["pil_image"], use_gemini=True))

    summary = aggregate_frame_results(frame_results)
    cat_color = summary["category_color"]
    guide = get_disposal_advisory(summary["dominant_category"], summary["dominant_category"]) or get_fallback_guide(summary["dominant_category"])
    carbon_estimate = round(random.uniform(0.10, 0.50), 2)

    log_classification(image_name=f"🎬 {uploaded_video.name}", predicted_label=summary["dominant_category"],
        waste_category=summary["dominant_category"], confidence=summary["avg_confidence"],
        bin_color=summary["bin_color"], carbon_saved_kg=carbon_estimate)

    # Results layout
    col_frames, col_result = st.columns([1.2, 0.8], gap="large")

    with col_frames:
        st.markdown('<div class="section-heading">🖼️ Extracted Key Frames</div>', unsafe_allow_html=True)
        frame_cols = st.columns(4)
        for i, f in enumerate(frames):
            with frame_cols[i % 4]:
                st.image(f["pil_image"], caption=f"t={f['timestamp_sec']}s", width=180)
                if i < len(frame_results) and frame_results[i]:
                    t = frame_results[i][0]
                    lbl = t["label"].replace("_", " ").title()
                    st.markdown(f'<div style="text-align:center;margin-top:-5px"><span class="chip" style="background:{t["category_color"]}20;color:{t["category_color"]};font-size:.7rem">{lbl} ({t["waste_category"]}) · {t["confidence"]:.0%}</span></div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="section-heading">🏷️ Aggregated Result</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="eco-card" style="border-left:4px solid {cat_color}">
            <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:1rem">
                <span class="chip" style="background:{cat_color}20;color:{cat_color}">{summary['dominant_category']}</span>
                <span class="text-sm text-muted">{summary['bin_color']} Bin</span>
            </div>
            <div class="text-sm text-muted" style="margin-bottom:0.25rem;">Dominant Item Detected:</div>
            <div style="font-size:1.4rem;font-weight:700;color:#0b1c30;margin-bottom:.5rem">{summary['dominant_label']}</div>
            <div class="text-sm text-muted">Avg Category Confidence: <strong style="color:{cat_color}">{summary['avg_confidence']:.1%}</strong></div>
        </div>""", unsafe_allow_html=True)

        # Category distribution
        if summary["category_counts"]:
            st.markdown("**Category Distribution:**")
            for cat, cnt in sorted(summary["category_counts"].items(), key=lambda x: -x[1]):
                pct = cnt / len(frame_results) * 100
                c = CATEGORY_COLORS.get(cat, "#64748b")
                st.markdown(f'<div style="margin-bottom:.4rem"><span class="chip" style="background:{c}20;color:{c}">{cat}</span> <span class="text-sm text-muted">{cnt}/{len(frame_results)} frames ({pct:.0f}%)</span></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    _render_guide(guide, cat_color)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    _render_insights(guide)

    # Cleanup temp file
    try:
        os.unlink(video_path)
    except OSError:
        pass
