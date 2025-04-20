import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import warnings
import sys
import logging
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

# Configure logging to print to terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Print to terminal instead of showing in the app
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
warnings.filterwarnings('ignore')  # Suppress other warnings

# Set page config
st.set_page_config(
    page_title="Knee OA Severity Classifier",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        font-size: 40px !important;
        font-weight: bold !important;
        color: #2a9d8f !important;
        text-align: center;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 24px !important;
        color: #264653 !important;
        margin-top: 20px !important;
    }
    .prediction-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .confidence-bar {
        height: 30px;
        border-radius: 5px;
        margin: 10px 0;
        background: linear-gradient(90deg, #e9c46a, #f4a261);
        display: flex;
        align-items: center;
        padding-left: 10px;
        color: white;
        font-weight: bold;
    }
    .upload-box {
        border: 2px dashed #2a9d8f;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        cursor: pointer;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #6c757d;
        font-size: 14px;
    }
    .stWarning {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_keras_model():
    try:
        # Update this path to your model location
        model = load_model('models/inception_resnet_best_model.keras')
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

model = load_keras_model()

# Class information
class_info = {
    0: {"name": "Normal", "desc": "No signs of osteoarthritis", "color": "#2a9d8f"},
    1: {"name": "Doubtful", "desc": "Possible minimal osteophytes, uncertain", "color": "#8ab17d"},
    2: {"name": "Mild", "desc": "Definite osteophytes, possible joint space narrowing", "color": "#e9c46a"},
    3: {"name": "Moderate", "desc": "Multiple osteophytes, definite joint space narrowing", "color": "#f4a261"},
    4: {"name": "Severe", "desc": "Large osteophytes, severe joint space narrowing, bone deformation", "color": "#e76f51"}
}

# Image preprocessing
def preprocess_img(img, target_size=(224, 224)):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.resize(img, target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Confidence plot function
def plot_confidence(predictions):
    fig, ax = plt.subplots(figsize=(8, 4))
    classes = [class_info[i]["name"] for i in range(5)]
    colors = [class_info[i]["color"] for i in range(5)]
    bars = ax.barh(classes, predictions[0], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence')
    ax.set_title('Prediction Confidence')
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}',
                ha='left', va='center')
    plt.tight_layout()
    return fig

# Main application
def main():
    st.markdown('<div class="header">Knee Osteoarthritis Severity Classifier</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        Upload a knee X-ray image to assess osteoarthritis severity using our AI model.
        The system will classify the image into one of 5 KL grades.
    </div>
    """, unsafe_allow_html=True)

    # File upload section
    uploaded_file = st.file_uploader("Upload Knee X-ray Image", 
                                   type=["jpg", "jpeg", "png"], 
                                   accept_multiple_files=False,
                                   help="Upload a knee X-ray image in JPG, JPEG, or PNG format")
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="subheader">Uploaded Image</div>', unsafe_allow_html=True)
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded X-ray", use_container_width=True)
            if st.button("Analyze Image", type="primary", use_container_width=True):
                if model is None:
                    st.error("Model could not be loaded. Please check the model path.")
                else:
                    with st.spinner("Analyzing image..."):
                        try:
                            start_time = time.time()
                            # Convert and preprocess image
                            img_array = np.array(img)
                            processed_img = preprocess_img(img_array)
                            # Make prediction
                            logger.info("Making prediction...")
                            with tf.device('/CPU:0'):  # Force CPU to avoid potential GPU issues
                                predictions = model.predict(processed_img, verbose=0)
                            pred_class = np.argmax(predictions[0])
                            confidence = predictions[0][pred_class]
                            logger.info(f"Predicted class: {pred_class} with confidence: {confidence:.4f}")
                            
                            end_time = time.time()
                            processing_time = end_time - start_time
                            logger.info(f"Total processing time: {processing_time:.2f}s")
                            
                            # Display results in the right column
                            with col2:
                                st.markdown('<div class="subheader">Analysis Results</div>', unsafe_allow_html=True)
                                # Prediction result box
                                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                                st.markdown(f"""
                                <h3 style='color: {class_info[pred_class]["color"]}; text-align: center;'>
                                    {class_info[pred_class]["name"]} (KL Grade {pred_class})
                                </h3>
                                <p style='text-align: center;'>{class_info[pred_class]["desc"]}</p>
                                <div class="confidence-bar" style="width: {confidence*100}%;">
                                    Confidence: {confidence:.2f}
                                </div>
                                <p style='text-align: right; font-size: 12px; color: #6c757d;'>
                                    Processing time: {processing_time:.2f}s
                                </p>
                                """, unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Confidence graph
                                st.markdown('<div class="subheader">Confidence Distribution</div>', unsafe_allow_html=True)
                                conf_fig = plot_confidence(predictions)
                                st.pyplot(conf_fig)
                                
                        except Exception as e:
                            logger.error(f"An error occurred during analysis: {str(e)}", exc_info=True)
                            st.error("An error occurred during the analysis. Please check the terminal for details.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <hr>
        <p>Knee Osteoarthritis Severity Classifier | AI Radiology Assistant</p>
        <p>This tool is for research purposes only. Always consult a medical professional for diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()