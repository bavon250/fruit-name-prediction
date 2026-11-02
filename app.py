
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Fruit Detection AI",
    page_icon="🍎",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.title {
    text-align: center;
    color: #167d45;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #666;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = Path("models/fruitdetection.pt")

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

# ---------------- HEADER ----------------
st.markdown(
    '<div class="title">🍎 Fruit Detection AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Intelligent Fruit Recognition '
    'Powered by YOLO11</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05
)

st.sidebar.info(
    "Upload a fruit image to detect Apple, "
    "Avocado, Banana and Orange."
)

# ---------------- MAIN APPLICATION ----------------
if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please place "
        "fruitdetection.pt inside the models folder."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "📤 Upload an image of fruits",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("🔍 Detect Fruits", type="primary",
                 use_container_width=True):

        with st.spinner("Detecting fruits..."):

            try:
                results = model.predict(
                    source=image,
                    conf=confidence,
                    verbose=False
                )

                result = results[0]

                annotated = result.plot()
                annotated = Image.fromarray(
                    annotated[:, :, ::-1]
                )

                with col2:
                    st.subheader("Detection Result")
                    st.image(
                        annotated,
                        use_container_width=True
                    )

                # Detection statistics
                detections = []

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    score = float(box.conf[0])

                    detections.append({
                        "Fruit": class_name,
                        "Confidence": f"{score:.2%}"
                    })

                st.divider()
                st.subheader("📊 Detection Statistics")

                if detections:

                    counts = Counter(
                        item["Fruit"] for item in detections
                    )

                    total = len(detections)

                    m1, m2 = st.columns(2)

                    m1.metric("Total Fruits Detected", total)
                    m2.metric("Different Fruit Types", len(counts))

                    st.subheader("Fruit Count")

                    for fruit, count in counts.items():
                        st.write(f"🍓 **{fruit}:** {count}")

                    st.subheader("Detection Details")
                    st.dataframe(
                        detections,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Download result
                    import io
                    buffer = io.BytesIO()
                    annotated.save(buffer, format="PNG")

                    st.download_button(
                        "⬇️ Download Detection Image",
                        data=buffer.getvalue(),
                        file_name="fruit_detection_result.png",
                        mime="image/png",
                        use_container_width=True
                    )

                else:
                    st.warning(
                        "No fruits detected. Try another image "
                        "or lower the confidence threshold."
                    )

            except Exception as e:
                st.error(f"Detection failed: {e}")

else:
    st.info(
        "👆 Please upload a fruit image to get started."
    )

st.divider()

st.markdown(
    "<center>Fruit Detection AI | Powered by YOLO11 "
    "and Streamlit</center>",
    unsafe_allow_html=True
)
