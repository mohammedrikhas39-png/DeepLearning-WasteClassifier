from pathlib import Path
import json
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATHS = [
    MODELS_DIR / "best_waste_classifier.keras",
    MODELS_DIR / "final_waste_classifier.keras",
    MODELS_DIR / "model.keras",
]
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

st.set_page_config(
    page_title="EcoSort AI - Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
   .main { background-color: #f7faf7; }
   .header { background: linear-gradient(135deg, #0F2C1F 0%, #1B5E20 100%); padding: 28px; border-radius: 14px; color: white; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
   .header h1 { color: #81C784; font-size: 42px; margin: 0; font-weight: 700; }
   .header p { color: #E8F5E9; font-size: 16px; margin-top: 8px; }
   .metric-card { background: white; padding: 22px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.06); border-left: 6px solid #2E7D32; margin-bottom: 16px; }
   .guidance-box { background: #E8F5E9; padding: 16px; border-radius: 10px; border: 1px solid #C8E6C9; margin-top: 14px; }
   .stButton>button { background: linear-gradient(90deg, #2E7D32 0%, #1B5E20 100%); color: white; border: none; border-radius: 8px; height: 50px; font-weight: 600; font-size: 16px; letter-spacing: 0.5px; width: 100%; transition: all 0.3s; }
   .stButton>button:hover { background: linear-gradient(90deg, #388E3C 0%, #2E7D32 100%); box-shadow: 0 4px 12px rgba(46,125,50,0.3); }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="header"><h1>♻️ EcoSort AI</h1><p>Smart Waste Classification System | Powered by Deep Learning & MobileNetV2</p></div>',
    unsafe_allow_html=True
)

DISPOSAL_GUIDELINES = {
    "cardboard": {
        "bin": "Blue Recycling Bin 📦",
        "action": "Flatten all boxes to save space. Remove tape and plastic packaging. Keep dry."
    },
    "glass": {
        "bin": "Glass / Green Recycling Bin 🍾",
        "action": "Empty and rinse bottles or jars. Remove metal lids and corks. Do not recycle broken window or mirror glass."
    },
    "metal": {
        "bin": "Yellow / Metal Recycling Bin 🥫",
        "action": "Rinse food and beverage cans. Labels can remain on. Aerosol cans must be empty."
    },
    "paper": {
        "bin": "Blue Paper Recycling Bin 📄",
        "action": "Keep clean and dry. Avoid recycling greasy pizza boxes, wax-coated paper, or soiled napkins."
    },
    "plastic": {
        "bin": "Yellow / Plastics Recycling Bin 🥤",
        "action": "Check recycling resin code (PETE 1, HDPE 2). Rinse thoroughly. Squash bottles to reduce volume."
    },
    "trash": {
        "bin": "Black / Landfill Waste Bin 🗑️",
        "action": "Non-recyclable general waste. Ensure safe disposal without contamination of recyclable bins."
    }
}

@st.cache_resource
def load_model():
    for path in MODEL_PATHS:
        if path.exists():
            try:
                loaded = tf.keras.models.load_model(str(path))
                return loaded, path.name
            except Exception as e:
                st.warning(f"Failed loading {path.name}: {e}")
    return None, None

@st.cache_data
def load_class_names():
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, 'r', encoding="utf-8") as f:
            return json.load(f)
    return ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

model, model_filename = load_model()
class_names = load_class_names()

if model is None:
    st.error(f"❌ No trained model found in `{MODELS_DIR}`. Please run `python src/train.py` first.")
    st.stop()

# Validate output units against loaded class names
expected_num_classes = model.output_shape[-1]
if len(class_names) != expected_num_classes:
    st.warning(
        f"⚠️ Warning: Model expects {expected_num_classes} classes, but class list has {len(class_names)}. "
        f"Please verify `models/class_names.json`."
    )

# Sidebar with system info
with st.sidebar:
    st.header("⚙️ Model Details")
    st.markdown(f"**Loaded Model:** `{model_filename}`")
    st.markdown(f"**Backbone:** `MobileNetV2` (Transfer Learning)")
    st.markdown(f"**Input Size:** `224 x 224 x 3`")
    st.markdown(f"**Classes ({len(class_names)}):**")
    for name in class_names:
        st.markdown(f"- `{name.capitalize()}`")
    
    st.divider()
    st.caption("EcoSort AI System v2.0 | Optimized for accurate, real-time waste sorting.")

# Initialize session state for prediction persistence
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "current_image_source" not in st.session_state:
    st.session_state["current_image_source"] = None

# Two-column layout
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.subheader("📤 Provide Waste Image")
    input_method = st.radio("Choose Input Method:", ["Upload File", "Use Camera"], horizontal=True)

    image = None
    source_identifier = None

    if input_method == "Upload File":
        uploaded = st.file_uploader(
            "Upload an image (JPG, PNG, JPEG, WEBP)",
            type=["jpg", "png", "jpeg", "webp"],
            label_visibility="collapsed"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            source_identifier = f"upload_{uploaded.name}_{uploaded.size}"
    else:
        camera_photo = st.camera_input("Capture Waste Photo", label_visibility="collapsed")
        if camera_photo:
            image = Image.open(camera_photo).convert("RGB")
            source_identifier = f"camera_{camera_photo.size}"

    if image is not None:
        st.image(image, use_container_width=True, caption="Source Image for Classification")
        st.caption(f"Categories: {', '.join([x.upper() for x in class_names])}")

        # Clear cached prediction if image changed
        if st.session_state["current_image_source"] != source_identifier:
            st.session_state["prediction_result"] = None
            st.session_state["current_image_source"] = source_identifier

        if st.button("🔍 CLASSIFY WASTE ITEM"):
            with st.spinner("Analyzing image features..."):
                # Resize image to model expected dimension
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)
                # NOTE: The model internally applies mobilenet_v2.preprocess_input.
                # Passing raw [0, 255] avoids double-normalization corruption.
                predictions = model.predict(img_array, verbose=0)[0]
                idx = int(np.argmax(predictions))
                confidence = float(predictions[idx] * 100.0)

                st.session_state["prediction_result"] = {
                    "predicted_class": class_names[idx] if idx < len(class_names) else f"Class #{idx}",
                    "confidence": confidence,
                    "probabilities": [float(p) for p in predictions],
                }

with right:
    st.subheader("📊 Classification Result")
    result = st.session_state.get("prediction_result")

    if result is None:
        st.info("👈 Upload an image or take a photo on the left, then click **Classify Waste Item** to see results.")
        st.markdown("""
        ### How EcoSort AI Works:
        1. **Deep Feature Extraction:** Evaluates textures, reflections, edges, and geometries via a deep neural network.
        2. **Multi-Class Probability:** Computes distribution across all supported waste categories.
        3. **Eco Guidance:** Provides immediate instructions on correct bin selection and preparation.
        """)
    else:
        pred_class = result["predicted_class"]
        conf = result["confidence"]
        probs = result["probabilities"]

        st.markdown(f"""
        <div class="metric-card">
            <p style='color:#666; margin:0; text-transform:uppercase; letter-spacing:1px; font-weight:600;'>Predicted Material</p>
            <h1 style='color:#1B5E20; margin:4px 0 0 0; font-size:38px;'>{pred_class.upper()}</h1>
            <h2 style='margin:4px 0 0 0; color:#2E7D32; font-size:28px;'>{conf:.2f}% <span style='font-size:16px; color:#777; font-weight:normal;'>Confidence</span></h2>
        </div>
        """, unsafe_allow_html=True)

        if conf < 60.0:
            st.warning("⚠️ **Low Confidence:** For higher accuracy, try taking a photo with plain background and better lighting.")

        # Disposal and recycling advice
        class_key = pred_class.lower()
        if class_key in DISPOSAL_GUIDELINES:
            guide = DISPOSAL_GUIDELINES[class_key]
            st.markdown(f"""
            <div class="guidance-box">
                <strong style='color:#1B5E20;'>Recommended Bin:</strong> {guide['bin']}<br/>
                <strong style='color:#1B5E20;'>Preparation Tip:</strong> {guide['action']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Probability Distribution")
        for i, name in enumerate(class_names):
            if i < len(probs):
                prob = probs[i]
                col_name, col_val = st.columns([0.7, 0.3])
                col_name.write(f"**{name.capitalize()}**")
                col_val.write(f"{prob * 100:.1f}%")
                clamped_val = min(max(float(prob), 0.0), 1.0)
                st.progress(clamped_val)