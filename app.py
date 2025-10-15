import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
from datetime import datetime
import time
import io
import matplotlib.pyplot as plt

# ========================================
# ⚙️ CONFIGURATION
# ========================================
MODEL_PATH = "weights/custom_yolov8.pt"
LOGO_URL = "images/chartscan.png"

st.set_page_config(
    page_title="📊 ChartScanAI - Intelligent Chart Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ========================================
# 🎨 MODERN STYLING
# ========================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    color: #fff;
    font-family: 'Poppins', sans-serif;
}
.hero-container {
    text-align: center;
    margin-top: 1em;
}
.hero-title {
    color: #F4D03F;
    font-size: 2.6em;
    font-weight: 700;
    text-shadow: 0 3px 15px rgba(255, 223, 77, 0.5);
}
.hero-subtext {
    color: #EAECEE;
    font-size: 1.1em;
    margin-bottom: 1.2em;
}
.result-card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    margin-top: 2em;
}
.stat-box {
    display: inline-block;
    background: linear-gradient(90deg, #2ECC71, #27AE60);
    color: #fff;
    border-radius: 10px;
    padding: 0.4em 1em;
    margin: 0.3em;
    font-weight: 600;
}
.timestamp {
    color: #BDC3C7;
    font-size: 0.85em;
    text-align: right;
    margin-top: 1em;
}
.feedback {
    background: rgba(255,255,255,0.07);
    padding: 1em;
    border-radius: 12px;
    margin-top: 2em;
}
</style>
""", unsafe_allow_html=True)

# ========================================
# 🧠 HERO HEADER
# ========================================
st.markdown("""
<div class='hero-container'>
    <div style='font-size:3em;'>📊</div>
    <div class='hero-title'>ChartScanAI</div>
    <div class='hero-subtext'>
        AI-powered candlestick pattern recognition — Upload any chart and let the model highlight BUY/SELL signals intelligently 🚀
    </div>
</div>
""", unsafe_allow_html=True)

# ========================================
# 📤 SIDEBAR SETTINGS
# ========================================
st.sidebar.image(LOGO_URL, use_container_width=True)
st.sidebar.markdown("### ⚙️ Detection Settings")
source_img = st.sidebar.file_uploader("📈 Upload Chart", type=["jpg", "jpeg", "png", "bmp", "webp"])
confidence = st.sidebar.slider("🎯 Confidence Threshold", 0.25, 1.0, 0.35, step=0.05)
detect_btn = st.sidebar.button("🔍 Run Detection")
show_chart = st.sidebar.checkbox("📊 Show Confidence Chart", True)
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by **ChartScanAI**")

# ========================================
# LOAD YOLO MODEL
# ========================================
@st.cache_resource
def load_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as ex:
        st.error(f"❌ Failed to load YOLO model: {ex}")
        return None

model = load_model()

# ========================================
# MAIN INTERFACE
# ========================================
col1, col2 = st.columns([1, 1.3])
if source_img:
    with col1:
        uploaded_image = Image.open(source_img)
        st.image(uploaded_image, caption="📥 Uploaded Chart", use_container_width=True)
else:
    st.warning("⚠️ Upload a candlestick chart image to start detection.")

# ========================================
# DETECTION LOGIC
# ========================================
if detect_btn and source_img and model:
    start_time = time.time()
    with st.spinner("Analyzing chart patterns... please wait 🧠"):
        try:
            source_img.seek(0)
            uploaded_image = Image.open(source_img).convert("RGB")
            img_array = np.array(uploaded_image)
            results = model.predict(uploaded_image, conf=confidence)
            detections = results[0].boxes
            plotted_img = img_array.copy()
            buy_count, sell_count = 0, 0
            confidences = []

            for box in detections:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0]) if box.cls is not None else -1
                conf_score = float(box.conf[0]) if box.conf is not None else 0
                confidences.append(conf_score)
                if cls == 0:
                    label, color = "BUY", (0, 255, 0)
                    buy_count += 1
                elif cls == 1:
                    label, color = "SELL", (255, 0, 0)
                    sell_count += 1
                else:
                    label, color = f"Class {cls}", (255, 255, 0)
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(plotted_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(plotted_img, f"{label} {conf_score:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Performance metrics
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)

            # ========================================
            # DISPLAY RESULTS
            # ========================================
            with col2:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.subheader("✅ Detection Results")
                st.image(plotted_img, caption="AI Detection Output", use_container_width=True)

                st.markdown(
                    f"""
                    <div class='stat-box'>🟢 BUY Signals: {buy_count}</div>
                    <div class='stat-box'>🔴 SELL Signals: {sell_count}</div>
                    <div class='stat-box'>⚡ Inference Time: {processing_time}s</div>
                    """,
                    unsafe_allow_html=True,
                )

                total = buy_count + sell_count
                if total > 0:
                    ratio = round((buy_count / total) * 100, 1)
                    st.metric("📊 BUY/SELL Ratio", f"{ratio}% BUY", delta=f"{100 - ratio}% SELL")

                if show_chart and len(confidences) > 0:
                    fig, ax = plt.subplots()
                    ax.hist(confidences, bins=10, color="#1ABC9C", alpha=0.8)
                    ax.set_title("Confidence Distribution", color="#F4D03F")
                    ax.set_xlabel("Confidence")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)

                # Generate DataFrame
                data = []
                for i, box in enumerate(detections):
                    cls = int(box.cls[0])
                    label = "BUY" if cls == 0 else "SELL"
                    conf = float(box.conf[0])
                    xywh = box.xywh.tolist()[0]
                    data.append([i + 1, label, f"{conf:.2f}", [round(v, 2) for v in xywh]])
                df = pd.DataFrame(data, columns=["#", "Label", "Confidence", "XYWH"])

                with st.expander("📋 Detection Data", expanded=False):
                    st.dataframe(df, use_container_width=True)

                    # Download CSV
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="💾 Download Results as CSV",
                        data=csv,
                        file_name=f"chartscan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

                # Smart insights
                st.markdown("<div class='section-title'>💡 Smart Insights</div>", unsafe_allow_html=True)
                if buy_count > sell_count:
                    st.success("Market shows strong BUY sentiment — possible upward trend 📈")
                elif sell_count > buy_count:
                    st.error("More SELL signals detected — potential bearish movement 📉")
                else:
                    st.info("Neutral sentiment — mixed signals detected ⚖️")

                st.markdown(f"<div class='timestamp'>🕒 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error("⚠️ Error during detection.")
            st.exception(e)
