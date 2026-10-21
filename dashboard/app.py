"""
SafeRide AI — Streamlit Dashboard 🏍️

Launch:
    streamlit run dashboard/app.py

Features:
  • Upload & process videos (or pick a sample)
  • Live-annotated frame viewer
  • Violation log table with search & filters
  • Statistics cards and charts
  • Crop image gallery
"""

from __future__ import annotations

import os
import sys
import glob
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ── Ensure project root on path ──────────────────────────
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config.settings as cfg

# ─────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title=cfg.DASHBOARD_PAGE_TITLE,
    page_icon=cfg.DASHBOARD_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
#  Custom CSS  — premium dark-glass look
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- metric cards ---------- */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30,35,50,0.85), rgba(20,24,38,0.95));
    border: 1px solid rgba(100,120,255,0.15);
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    backdrop-filter: blur(12px);
}

div[data-testid="stMetric"] label {
    color: rgba(180,190,220,0.85) !important;
    font-weight: 500;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6C8CFF, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1117 0%, #161b22 100%);
    border-right: 1px solid rgba(100,120,255,0.08);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #A78BFA;
}

/* ---------- header area ---------- */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6C8CFF, #A78BFA, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: rgba(180,190,220,0.7);
    margin-top: 4px;
    margin-bottom: 28px;
}

/* ---------- dataframe ---------- */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* ---------- tabs ---------- */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.95rem;
}

/* ---------- upload area ---------- */
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(100,120,255,0.25);
    border-radius: 14px;
    padding: 16px;
    transition: border-color 0.3s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(100,120,255,0.5);
}

/* ---------- violation badge ---------- */
.badge-no-helmet {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-triple {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* ---------- divider ---------- */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(100,120,255,0.4), transparent);
    margin: 20px 0;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🏍️ SafeRide AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    'Automated detection of helmetless riders &amp; triple riding from traffic footage'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    mode = st.radio(
        "Mode",
        ["📹 Process Video", "📊 Review Violations"],
        index=0,
    )

    st.markdown("---")

    if mode == "📹 Process Video":
        st.markdown("### Detection Options")
        confidence = st.slider(
            "YOLO Confidence", 0.1, 0.9, cfg.YOLO_CONFIDENCE, 0.05
        )
        skip_frames = st.slider(
            "Skip Frames (0 = all)", 0, 10, 0
        )
        enable_ocr = st.checkbox("Enable Plate OCR", value=True)
        max_frames = st.number_input(
            "Max Frames (0 = all)", min_value=0, value=0, step=100
        )
        save_output = st.checkbox("Save Annotated Video", value=True)

    st.markdown("---")
    st.markdown(
        "**SafeRide AI** v1.0  \n"
        "YOLOv8 · DeepSORT · MobileNetV2 · EasyOCR"
    )


# ─────────────────────────────────────────────────────────
#  Helper: load violations CSV
# ─────────────────────────────────────────────────────────
def load_violations() -> pd.DataFrame:
    if not os.path.exists(cfg.VIOLATIONS_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(cfg.VIOLATIONS_CSV)
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────
#  MODE 1: Process Video
# ─────────────────────────────────────────────────────────
if mode == "📹 Process Video":
    uploaded = st.file_uploader(
        "Upload a traffic video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload traffic camera footage to analyse.",
    )

    if uploaded is not None:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()
        input_path = tfile.name

        st.video(uploaded)
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            # Update config dynamically
            cfg.YOLO_CONFIDENCE = confidence

            from src.pipeline.processor import SafeRidePipeline

            output_path = None
            if save_output:
                output_path = os.path.join(cfg.OUTPUT_DIR, "annotated_output.mp4")

            with st.spinner("🔄 Loading models (YOLOv8, DeepSORT, EasyOCR)… this may take a moment on first run"):
                pipeline = SafeRidePipeline(enable_ocr=enable_ocr)

            # Progress tracking
            cap = cv2.VideoCapture(input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            if max_frames > 0:
                total_frames = min(total_frames, max_frames)

            progress = st.progress(0, text="Initialising pipeline…")
            status_text = st.empty()
            frame_display = st.empty()
            violations_placeholder = st.empty()

            # Re-open video and process frame by frame
            cap = cv2.VideoCapture(input_path)
            writer = None
            if output_path:
                from src.utils.io_helpers import VideoWriter
                writer = VideoWriter(output_path, fps, w, h)

            frame_idx = 0
            all_violations: list[dict] = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames > 0 and frame_idx >= max_frames:
                    break

                if skip_frames and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                # Update progress BEFORE processing so user sees current frame
                pct = (frame_idx + 1) / total_frames if total_frames else 0
                progress.progress(
                    min(pct, 1.0),
                    text=f"Processing frame {frame_idx + 1}/{total_frames}…",
                )

                timestamp = frame_idx / fps
                annotated, new_v = pipeline.process_frame(frame, frame_idx, timestamp)
                all_violations.extend(new_v)

                if writer:
                    writer.write(annotated)

                # Update live preview every 3 frames (balance speed vs feedback)
                if frame_idx % 3 == 0:
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_display.image(rgb, channels="RGB", use_container_width=True)

                    status_text.markdown(
                        f"**Violations found so far:** {len(all_violations)}"
                    )

                frame_idx += 1

            cap.release()
            if writer:
                writer.release()

            progress.progress(1.0, text="✅ Complete!")

            st.success(
                f"Processed {frame_idx} frames — "
                f"found **{len(all_violations)}** violations."
            )

            # Clean up temp
            os.unlink(input_path)

            # Show results
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📊 Results")

            # Metric cards
            c1, c2, c3, c4 = st.columns(4)
            n_helmet = sum(1 for v in all_violations if v["type"] == "no_helmet")
            n_triple = sum(1 for v in all_violations if v["type"] == "triple_riding")
            n_plates = sum(1 for v in all_violations if v.get("plate_text"))

            c1.metric("Total Violations", len(all_violations))
            c2.metric("No Helmet", n_helmet)
            c3.metric("Triple Riding", n_triple)
            c4.metric("Plates Read", n_plates)

            # Violations table
            if all_violations:
                vdf = pd.DataFrame([{
                    "ID": v.get("violation_id", ""),
                    "Time": f"{v.get('timestamp', 0):.1f}s",
                    "Type": v["type"].replace("_", " ").title(),
                    "Plate": v.get("plate_text") or "—",
                    "Track ID": v.get("track_id", "—"),
                } for v in all_violations])
                st.dataframe(vdf, use_container_width=True, hide_index=True)

            if output_path and os.path.exists(output_path):
                st.markdown("### 🎬 Annotated Video")
                st.video(output_path)

    else:
        st.info("👆 Upload a traffic video to begin analysis.")


# ─────────────────────────────────────────────────────────
#  MODE 2: Review Violations
# ─────────────────────────────────────────────────────────
elif mode == "📊 Review Violations":
    df = load_violations()

    if df.empty:
        st.info("No violations recorded yet. Process a video first!")
    else:
        # Metric cards
        st.markdown("### 📈 Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Violations", len(df))
        if "violation_type" in df.columns:
            c2.metric(
                "No Helmet",
                len(df[df["violation_type"] == "no_helmet"]),
            )
            c3.metric(
                "Triple Riding",
                len(df[df["violation_type"] == "triple_riding"]),
            )
        if "plate_text" in df.columns:
            c4.metric(
                "Plates Read",
                df["plate_text"].notna().sum()
                - (df["plate_text"] == "").sum(),
            )

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # Tabs: table, crops, charts
        tab1, tab2, tab3 = st.tabs(["📋 Violation Log", "🖼️ Image Crops", "📊 Charts"])

        with tab1:
            # Filters
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                if "violation_type" in df.columns:
                    types = ["All"] + df["violation_type"].unique().tolist()
                    sel_type = st.selectbox("Filter by type", types)
            with col_f2:
                search = st.text_input("Search plate number")

            filtered = df.copy()
            if "violation_type" in df.columns and sel_type != "All":
                filtered = filtered[filtered["violation_type"] == sel_type]
            if search and "plate_text" in filtered.columns:
                filtered = filtered[
                    filtered["plate_text"]
                    .fillna("")
                    .str.contains(search.upper(), case=False)
                ]

            st.dataframe(filtered, use_container_width=True, hide_index=True)

            # CSV download
            csv_data = filtered.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv_data,
                file_name="saferide_violations.csv",
                mime="text/csv",
            )

        with tab2:
            st.markdown("### Violation Crops")
            crops = sorted(glob.glob(os.path.join(cfg.CROPS_DIR, "*.jpg")))
            if not crops:
                st.info("No crop images found.")
            else:
                cols = st.columns(4)
                for i, crop_path in enumerate(crops[:20]):
                    with cols[i % 4]:
                        img = cv2.imread(crop_path)
                        if img is not None:
                            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            st.image(rgb, caption=os.path.basename(crop_path),
                                     use_container_width=True)

        with tab3:
            st.markdown("### Violation Distribution")
            if "violation_type" in df.columns:
                import plotly.express as px

                # Type distribution
                type_counts = (
                    df["violation_type"]
                    .value_counts()
                    .reset_index()
                )
                type_counts.columns = ["Violation Type", "Count"]
                type_counts["Violation Type"] = (
                    type_counts["Violation Type"]
                    .str.replace("_", " ")
                    .str.title()
                )

                fig = px.bar(
                    type_counts,
                    x="Violation Type",
                    y="Count",
                    color="Violation Type",
                    color_discrete_sequence=["#ef4444", "#f59e0b", "#6C8CFF"],
                    title="Violations by Type",
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="rgba(200,210,230,0.9)",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Timeline if we have frame_idx
                if "frame_idx" in df.columns:
                    fig2 = px.scatter(
                        df,
                        x="frame_idx",
                        y="violation_type",
                        color="violation_type",
                        title="Violations Over Time (by frame)",
                        color_discrete_sequence=["#ef4444", "#f59e0b"],
                    )
                    fig2.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="rgba(200,210,230,0.9)",
                        yaxis_title="Type",
                        xaxis_title="Frame Index",
                    )
                    st.plotly_chart(fig2, use_container_width=True)
