import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

def _image_bytes(image):
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

st.set_page_config(
    page_title="Fruit Detection",
    page_icon="🍎",
    layout="wide",
)

st.title("🍎 Fruit Detection")
st.write("Upload an image or take a webcam picture to detect fruits using your trained RF-DETR model.")

@st.cache_resource(show_spinner="Loading the fruit detection model...")
def load_model(weights_path: str):
    # RF-DETR checkpoints are trusted here because the user supplies their own trained weights.
    from rfdetr import RFDETRSmall
    return RFDETRSmall(pretrain_weights=weights_path, trust_checkpoint=True)

def get_labels(detections, fallback_text):
    names = detections.data.get("class_name") if hasattr(detections, "data") else None
    if names is not None:
        return [str(x) for x in names]
    fallback = [x.strip() for x in fallback_text.split(",") if x.strip()]
    ids = getattr(detections, "class_id", [])
    return [fallback[int(i)] if fallback and int(i) < len(fallback) else f"Class {int(i)}" for i in ids]

def draw_detections(image, detections, fallback_text):
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    boxes = getattr(detections, "xyxy", [])
    scores = getattr(detections, "confidence", [])
    class_ids = getattr(detections, "class_id", [])
    labels = get_labels(detections, fallback_text)

    count = min(len(boxes), len(scores), len(class_ids), len(labels))
    rows = []
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for i in range(count):
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        score = float(scores[i])
        name = labels[i]
        caption = f"{name} {score:.1%}"
        draw.rectangle((x1, y1, x2, y2), outline=(0, 220, 80), width=4)
        bbox = draw.textbbox((x1, max(0, y1 - 23)), caption, font=font)
        draw.rectangle(bbox, fill=(0, 120, 45))
        draw.text((x1, max(0, y1 - 23)), caption, fill="white", font=font)
        rows.append({"Fruit / class": name, "Confidence": round(score * 100, 2)})
    return output, rows

st.sidebar.header("Model settings")
weights_upload = st.sidebar.file_uploader(
    "Upload trained weights (.pt)",
    type=["pt", "pth"],
    help="Choose your Roboflow-trained weights.pt file. The file is processed by this app and is not committed to GitHub.",
)
local_weights = Path("weights.pt")
threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.40, 0.05)
class_names = st.sidebar.text_input(
    "Class names (optional)",
    value="",
    help="Only needed if class names are not included in the checkpoint. Enter names in class-ID order, separated by commas.",
)

weights_path = None
if weights_upload is not None:
    suffix = Path(weights_upload.name).suffix or ".pt"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(weights_upload.getvalue())
    tmp.flush()
    tmp.close()
    weights_path = tmp.name
elif local_weights.exists():
    weights_path = str(local_weights.resolve())

if weights_path is None:
    st.info("Upload your trained weights.pt file in the left sidebar to begin. For local use, you can also place weights.pt beside app.py.")
else:
    try:
        model = load_model(weights_path)
        st.success("Model loaded. Choose an image or use the webcam below.")
        tab_image, tab_camera = st.tabs(["📁 Upload image", "📷 Webcam"])
        with tab_image:
            image_file = st.file_uploader("Choose a fruit image", type=["jpg", "jpeg", "png", "webp"], key="image_upload")
            image = Image.open(image_file).convert("RGB") if image_file else None
        with tab_camera:
            camera_file = st.camera_input("Take a picture")
            camera_image = Image.open(camera_file).convert("RGB") if camera_file else None

        selected_image = image if image is not None else camera_image
        if selected_image is not None:
            st.image(selected_image, caption="Input image", use_container_width=True)
            if st.button("Detect fruits", type="primary"):
                with st.spinner("Running detection..."):
                    detections = model.predict(selected_image, threshold=threshold)
                    annotated, rows = draw_detections(selected_image, detections, class_names)
                left, right = st.columns(2)
                with left:
                    st.subheader("Detection result")
                    st.image(annotated, use_container_width=True)
                    st.download_button(
                        "Download annotated image",
                        data=_image_bytes(annotated),
                        file_name="fruit_detection_result.png",
                        mime="image/png",
                    )
                with right:
                    st.subheader("Detected objects")
                    if rows:
                        st.dataframe(rows, use_container_width=True, hide_index=True)
                        st.metric("Total detections", len(rows))
                        csv = "Fruit / class,Confidence (%)\n" + "".join(
                            f'"{r["Fruit / class"]}",{r["Confidence"]}\n' for r in rows
                        )
                        st.download_button("Download results CSV", csv, "fruit_detections.csv", "text/csv")
                    else:
                        st.warning("No objects were detected above the selected confidence threshold.")
    except Exception as exc:
        st.error("The model could not be loaded or inference failed.")
        st.exception(exc)
        st.markdown(
            "Check that the uploaded checkpoint is the RF-DETR `weights.pt` file from your Roboflow training run. "
            "If the checkpoint architecture is not RF-DETR Small, the loader may need to be adjusted."
        )

st.divider()
st.caption("Fruit Detection • Streamlit + RF-DETR")
