import streamlit as st
import numpy as np
from PIL import Image
import base64
from Models import detection

# ─── Constants ───────────────────────────────────────────────────────────────

COLOR_SAMPLES = [
    (255, 255, 0),    # Yellow
    (0, 255, 255),    # Cyan
    (255, 200, 100),  # Light orange
    (180, 255, 180),  # Pale green
    (255, 128, 255),  # Light magenta
]

MODEL_DATA = {
    "Microplastics": {
        "class_names": ["fiber", "film", "fragment", "pallet"],
        "weights_name": "best.pt",
    }
}

CLASS_ICONS = {
    "fiber": "🧵",
    "film": "🎞️",
    "fragment": "🔬",
    "pallet": "🔩",
}

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Microplastics Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Background Image ────────────────────────────────────────────────────────

def get_base64_image(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

bg_encoded = get_base64_image("img.jpg")
bg_style = (
    f'background-image: url("data:image/jpg;base64,{bg_encoded}");'
    if bg_encoded
    else "background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #112240 100%);"
)

# ─── Global CSS ──────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Root & App Background ── */
    .stApp {{
        {bg_style}
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        font-family: 'Space Grotesk', sans-serif;
    }}

    /* Overlay to darken background for readability */
    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        background: rgba(5, 10, 20, 0.72);
        z-index: 0;
        pointer-events: none;
    }}

    /* Ensure content sits above overlay */
    .block-container {{
        position: relative;
        z-index: 1;
        padding-top: 2rem !important;
        max-width: 1300px !important;
    }}

    /* ── Hero Header ── */
    .hero-wrapper {{
        text-align: center;
        padding: 2.5rem 1rem 2rem;
    }}
    .hero-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #00d4ff;
        margin-bottom: 0.75rem;
    }}
    .hero-title {{
        font-size: clamp(2rem, 5vw, 3.4rem);
        font-weight: 700;
        color: #f0f4ff;
        line-height: 1.15;
        margin: 0 0 0.6rem;
        letter-spacing: -0.02em;
    }}
    .hero-title span {{
        color: #00d4ff;
    }}
    .hero-subtitle {{
        font-size: 1rem;
        color: #7a8ca8;
        font-weight: 400;
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.6;
    }}

    /* ── Divider ── */
    .hero-divider {{
        width: 56px;
        height: 3px;
        background: linear-gradient(90deg, #00d4ff, #0077ff);
        border-radius: 2px;
        margin: 1.5rem auto;
    }}

    /* ── Control Panel Card ── */
    .control-card {{
        background: rgba(15, 25, 45, 0.75);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1.4rem;
    }}
    .control-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #00d4ff;
        margin-bottom: 0.5rem;
        display: block;
    }}

    /* ── Image panels ── */
    .img-panel {{
        background: rgba(10, 18, 35, 0.8);
        border: 1px solid rgba(0, 212, 255, 0.12);
        border-radius: 14px;
        overflow: hidden;
        backdrop-filter: blur(8px);
    }}
    .img-panel-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #4a6080;
        padding: 0.8rem 1rem 0.4rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }}

    /* ── Stats card ── */
    .stats-card {{
        background: rgba(10, 18, 35, 0.85);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }}
    .stats-title {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #00d4ff;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(0,212,255,0.15);
        padding-bottom: 0.6rem;
    }}
    .stat-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }}
    .stat-row:last-child {{ border-bottom: none; }}
    .stat-name {{
        font-size: 0.92rem;
        color: #c5d0e0;
        font-weight: 500;
    }}
    .stat-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: #f0f4ff;
    }}
    .stat-value.nonzero {{ color: #00d4ff; }}

    /* ── Confidence list ── */
    .conf-row {{
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.35rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #7a9ab8;
    }}
    .conf-bar-wrap {{
        flex: 1;
        height: 4px;
        background: rgba(255,255,255,0.07);
        border-radius: 2px;
        overflow: hidden;
    }}
    .conf-bar-fill {{
        height: 100%;
        background: linear-gradient(90deg, #0077ff, #00d4ff);
        border-radius: 2px;
    }}

    /* ── Run button ── */
    div.stButton > button {{
        background: linear-gradient(135deg, #0055cc 0%, #0099dd 100%);
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s, transform 0.15s;
    }}
    div.stButton > button:hover {{
        opacity: 0.9;
        transform: translateY(-1px);
    }}
    div.stButton > button:active {{
        transform: translateY(0);
    }}

    /* ── Selectbox ── */
    .stSelectbox label {{ color: #7a8ca8 !important; }}
    .stSelectbox > div > div {{
        background: rgba(10,18,35,0.8) !important;
        border: 1px solid rgba(0,212,255,0.2) !important;
        border-radius: 8px !important;
        color: #f0f4ff !important;
    }}

    /* ── File uploader ── */
    .stFileUploader label {{ color: #7a8ca8 !important; }}
    .stFileUploader > div {{
        background: rgba(10,18,35,0.7) !important;
        border: 1px dashed rgba(0,212,255,0.25) !important;
        border-radius: 10px !important;
    }}

    /* ── Alert / Error ── */
    .stAlert {{
        border-radius: 10px !important;
        border: 1px solid rgba(255,80,80,0.3) !important;
        background: rgba(80,10,10,0.5) !important;
    }}

    /* ── Spinner ── */
    .stSpinner > div {{ border-top-color: #00d4ff !important; }}

    /* ── Total badge ── */
    .total-badge {{
        display: inline-block;
        background: linear-gradient(135deg, #0055cc, #00aadd);
        color: #fff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        margin-bottom: 1rem;
    }}

    /* ── Placeholder panel ── */
    .placeholder-panel {{
        background: rgba(10,18,35,0.6);
        border: 1px dashed rgba(0,212,255,0.15);
        border-radius: 14px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 260px;
        color: #3a5070;
        font-size: 0.88rem;
        gap: 0.5rem;
        text-align: center;
        padding: 2rem;
    }}
    .placeholder-icon {{
        font-size: 2.4rem;
        opacity: 0.4;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Hero ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-wrapper">
        <div class="hero-eyebrow">YOLO26 · Computer Vision</div>
        <h1 class="hero-title">Microplastics <span>Detection</span></h1>
        <div class="hero-divider"></div>
        <p class="hero-subtitle">
            Upload an image to identify and count microplastic types —
            fiber, film, fragment, and pallet — using a custom-trained YOLO model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Controls ────────────────────────────────────────────────────────────────

ctrl_col, _, _ = st.columns([1.4, 1, 1])

with ctrl_col:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    selected_model_key = st.selectbox(
        "Model",
        list(MODEL_DATA.keys()),
        label_visibility="visible",
    )
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )
    st.markdown("</div>", unsafe_allow_html=True)

model_data = MODEL_DATA[selected_model_key]

# ─── Main Layout ─────────────────────────────────────────────────────────────

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    result_img = None
    counter_list = None
    scores = None

    col_orig, col_mid, col_result = st.columns([1.2, 0.85, 1.2], gap="large")

    # Original image
    with col_orig:
        st.markdown('<div class="img-panel">', unsafe_allow_html=True)
        st.markdown('<div class="img-panel-label">📥 Input Image</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Controls + stats
    with col_mid:
        run_btn = st.button("🚀 Run Detection", use_container_width=True)

        if run_btn:
            with st.spinner("Analysing image…"):
                try:
                    result_img, counter_list, scores = detection(
                        img,
                        model_data["class_names"],
                        COLOR_SAMPLES,
                        model_data["weights_name"],
                    )
                    st.session_state["result_img"] = result_img
                    st.session_state["counter_list"] = counter_list
                    st.session_state["scores"] = scores
                except Exception as e:
                    st.error(f"Detection failed: {e}")

        # Retrieve from session in case button wasn't just clicked
        result_img = st.session_state.get("result_img")
        counter_list = st.session_state.get("counter_list")
        scores = st.session_state.get("scores")

        if counter_list is not None:
            total = sum(counter_list)
            st.markdown(
                f'<div class="total-badge">Total detected: {total}</div>',
                unsafe_allow_html=True,
            )

            # Class counts
            rows_html = ""
            for i, name in enumerate(model_data["class_names"]):
                count = counter_list[i] if i < len(counter_list) else 0
                icon = CLASS_ICONS.get(name, "●")
                val_class = "nonzero" if count > 0 else ""

                # Each row inside the same box
                st.markdown(
                    f"""
                       <div class="stat-row">
                           <span class="stat-name">{icon} {name}</span>
                           <span class="stat-value {val_class}">{count}</span>
                       </div>
                       """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)


            # Confidence scores
            if scores:
                st.markdown('<div class="stats-card"><div class="stats-title">🎯 Confidence Scores</div>',
                            unsafe_allow_html=True)
                for idx, s in enumerate(scores[:12]):  # cap at 12 for readability
                    pct = int(s * 100)
                    st.markdown(
                        f"""
                        <div class="conf-row">
                            <span style="min-width:24px;text-align:right;">#{idx + 1}</span>
                            <div class="conf-bar-wrap">
                                <div class="conf-bar-fill" style="width:{pct}%"></div>
                            </div>
                            <span style="min-width:40px;">{s:.2f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                if len(scores) > 12:
                    st.markdown(
                        f'<div style="color:#3a5070;font-size:0.75rem;padding-top:0.3rem;">+{len(scores) - 12} more detections</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

    # Result image
    with col_result:
        if result_img is not None:
            st.markdown('<div class="img-panel">', unsafe_allow_html=True)
            st.markdown('<div class="img-panel-label">📤 Detection Output</div>', unsafe_allow_html=True)
            st.image(np.array(result_img), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="placeholder-panel">
                    <span class="placeholder-icon">🔬</span>
                    <span>Detection output will appear here</span>
                    <span style="font-size:0.78rem;opacity:0.6;">Press Run Detection to start</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

else:
    # No file uploaded — show hint panel
    st.markdown(
        """
        <div class="placeholder-panel" style="min-height:320px;margin-top:1rem;">
            <span class="placeholder-icon">📂</span>
            <span style="font-size:1rem;color:#4a6080;">Upload an image to get started</span>
            <span style="font-size:0.82rem;opacity:0.5;">Supported formats: JPG · JPEG · PNG</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clear any stale session results when a new session starts
    for key in ["result_img", "counter_list", "scores"]:
        st.session_state.pop(key, None)