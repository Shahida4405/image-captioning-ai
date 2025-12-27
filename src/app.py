import os
import sys
import warnings
import logging
import contextlib
import streamlit as st
from PIL import Image
import torch
from transformers.models.blip import BlipProcessor, BlipForConditionalGeneration

# ------------------------------------------------
# 🔹 Page Configuration (Dark Theme)
# ------------------------------------------------
st.set_page_config(
    page_title="AI Image Captioning Chatbot",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Apply Dark Theme via Custom CSS
st.markdown("""
    <style>
        /* Dark background */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        /* Upload section */
        .stFileUploader {
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 10px;
        }
        /* Success / Info messages */
        div.stAlert {
            background-color: #1E1E1E !important;
            border: 1px solid #3C3C3C;
            color: #FAFAFA !important;
        }
        /* Caption box */
        .caption-box {
            background-color: #262730;
            padding: 15px;
            border-radius: 10px;
            color: #FFFFFF;
            font-size: 1.1rem;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# 🔹 Configure Logging & Suppress Warnings
# ------------------------------------------------
logging.getLogger().handlers.clear()
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------
# 🔹 Load Model and Processor (from local directory)
# ------------------------------------------------
@st.cache_resource
def load_resources():
    st.info("🔄 Loading BLIP model from local folder... please wait ⏳")

    model_dir = r"C:\Users\shahi\image_captioning_chatbot\models\blip"

    try:
        processor = BlipProcessor.from_pretrained(model_dir)
        model = BlipForConditionalGeneration.from_pretrained(model_dir)
        st.success("✅ BLIP model loaded successfully from local folder!")
        return model, processor
    except Exception as e:
        st.error(f"❌ Failed to load BLIP model locally: {e}")
        st.stop()

# ------------------------------------------------
# 🔹 Caption Generation Function
# ------------------------------------------------
def generate_caption(model, processor, img):
    try:
        inputs = processor(images=img, return_tensors="pt")
        output = model.generate(**inputs, max_length=30, num_beams=5)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        raise RuntimeError(f"Caption generation failed: {e}")

# ------------------------------------------------
# 🔹 Streamlit App UI
# ------------------------------------------------
st.title("🖼️ AI Image Captioning Chatbot")
st.write("Upload an image and get a **realistic and intelligent caption instantly!**")

# Load model and processor once
model, processor = load_resources()

# File upload
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_file = Image.open(uploaded_file).convert("RGB")
    st.image(image_file, caption="📸 Uploaded Image", use_container_width=True)
    st.write("⏳ Generating caption...")

    try:
        caption = generate_caption(model, processor, image_file)
        st.success("✨ Caption Generated:")
        st.markdown(f"<div class='caption-box'>💬 {caption.capitalize()}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👆 Please upload an image to begin.")
