import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Streamlit page config must come first
st.set_page_config(page_title="🚦 Object Detection", layout="centered")

# Path to your trained YOLOv8 model
MODEL_PATH = r"C:\Users\VM MANIKANDAN\Downloads\HARSHU 2\HARSHU 2\harshu_yolov8_final.pt"

# Load the model
try:
    model = YOLO(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

st.title("🚦 Object Detection using YOLOv8")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Predict using YOLOv8
        with st.spinner("Detecting objects..."):
            # Convert PIL image to format expected by YOLO
            results = model.predict(image, conf=0.25, device='cpu')  # Using CPU since your training was on CPU
            # Convert the result to PIL Image for display
            result_image = Image.fromarray(results[0].plot())

        # Show result
        st.image(result_image, caption="Detected Image", use_container_width=True)
        st.success("✅ Detection complete!")

        # Optional: Show detection details
        detections = results[0].boxes
        st.write(f"Number of detections: {len(detections)}")
        for box in detections:
            class_name = results[0].names[int(box.cls)]
            confidence = float(box.conf)
            st.write(f"Detected: {class_name} (Confidence: {confidence:.2f})")

    except Exception as e:
        st.error(f"❌ Error during detection: {str(e)}")

# Optional: Add some info about the model
st.sidebar.header("About")
st.sidebar.info("""
This app uses a YOLOv8 model trained on custom data.
Upload an image to detect objects using the trained model.
""")