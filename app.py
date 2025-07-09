import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from utils import preprocess_image, greedy_generator, beam_search_generator
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Image Captioning App",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Custom CSS Styling ---
st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
        
        /* Main App Styling with New Background */
        .main {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #533483 100%);
            background-attachment: fixed;
            font-family: 'Inter', sans-serif;
            color: white;
            position: relative;
            overflow-x: hidden;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #533483 100%);
            background-attachment: fixed;
        }
        
        /* Add animated background particles */
        .main::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.1) 0%, transparent 50%),
                        radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.1) 0%, transparent 50%),
                        radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.1) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }
        
        /* Title Container with Glassmorphism */
        .title-container {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(25px);
            border-radius: 25px;
            padding: 50px;
            margin: 30px 0;
            text-align: center;
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.2),
                        0 0 0 1px rgba(255, 255, 255, 0.1),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .title-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: conic-gradient(from 0deg, transparent, rgba(120, 119, 198, 0.2), transparent);
            animation: rotate 20s linear infinite;
            z-index: -1;
        }
        
        @keyframes rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .title-container h1 {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #ffeaa7);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 15px;
            animation: gradientShift 4s ease-in-out infinite;
        }
        
        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        .subtitle {
            font-size: 1.3rem;
            color: rgba(255, 255, 255, 0.9);
            margin-top: 15px;
            font-weight: 400;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }
        
        /* Enhanced Container Styling */
        .upload-container, .model-info, .feature-card {
            background: rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 35px;
            margin: 25px 0;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15),
                        0 0 0 1px rgba(255, 255, 255, 0.08),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.08);
            transition: all 0.3s ease;
        }
        
        .upload-container:hover, .model-info:hover, .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.2),
                        0 0 0 1px rgba(255, 255, 255, 0.15),
                        inset 0 1px 0 rgba(255, 255, 255, 0.15);
        }
        
        /* Image Preview Container */
        .image-preview {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 25px;
            margin: 25px 0;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15),
                        0 0 0 1px rgba(255, 255, 255, 0.08),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.08);
            transition: all 0.3s ease;
        }
        
        .image-preview:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.2);
        }
        
        /* Caption Results Container */
        .caption-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(25px);
            border-radius: 25px;
            padding: 40px;
            margin: 30px 0;
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.2),
                        0 0 0 1px rgba(255, 255, 255, 0.1),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .caption-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            animation: shimmer 3s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        /* Enhanced Caption Text Styling */
        .greedy-caption {
            background: linear-gradient(135deg, #ff6b6b, #ff8e8e, #ffa8a8);
            background-size: 200% 200%;
            padding: 25px;
            border-radius: 18px;
            margin: 20px 0;
            font-size: 1.2rem;
            font-weight: 500;
            color: white;
            box-shadow: 0 8px 30px rgba(255, 107, 107, 0.4),
                        inset 0 1px 0 rgba(255, 255, 255, 0.2);
            animation: gradientMove 6s ease-in-out infinite;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .beam-caption {
            background: linear-gradient(135deg, #4ecdc4, #44a08d, #56c5b0);
            background-size: 200% 200%;
            padding: 25px;
            border-radius: 18px;
            margin: 20px 0;
            font-size: 1.2rem;
            font-weight: 500;
            color: white;
            box-shadow: 0 8px 30px rgba(78, 205, 196, 0.4),
                        inset 0 1px 0 rgba(255, 255, 255, 0.2);
            animation: gradientMove 6s ease-in-out infinite;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        @keyframes gradientMove {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        /* Enhanced Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
            background-size: 200% 200%;
            color: white;
            border: none;
            border-radius: 15px;
            padding: 18px 35px;
            font-size: 1.1rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4),
                        inset 0 1px 0 rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            animation: gradientMove 4s ease-in-out infinite;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5),
                        inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }
        
        .stButton > button:active {
            transform: translateY(-1px);
        }
        
        /* File Uploader Styling */
        .uploadedFile {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Enhanced Progress and Status Messages */
        .stInfo {
            background: rgba(78, 205, 196, 0.15);
            backdrop-filter: blur(10px);
            border-left: 4px solid #4ecdc4;
            border-radius: 12px;
            padding: 18px;
            margin: 15px 0;
            border: 1px solid rgba(78, 205, 196, 0.3);
        }
        
        .stSuccess {
            background: rgba(46, 204, 113, 0.15);
            backdrop-filter: blur(10px);
            border-left: 4px solid #2ecc71;
            border-radius: 12px;
            padding: 18px;
            margin: 15px 0;
            border: 1px solid rgba(46, 204, 113, 0.3);
        }
        
        .stError {
            background: rgba(231, 76, 60, 0.15);
            backdrop-filter: blur(10px);
            border-left: 4px solid #e74c3c;
            border-radius: 12px;
            padding: 18px;
            margin: 15px 0;
            border: 1px solid rgba(231, 76, 60, 0.3);
        }
        
        /* Enhanced Loading Animation */
        .loading-container {
            text-align: center;
            padding: 40px;
        }
        
        .loading-spinner {
            border: 4px solid rgba(255, 255, 255, 0.2);
            border-top: 4px solid #ff6b6b;
            border-right: 4px solid #4ecdc4;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 25px auto;
            box-shadow: 0 0 20px rgba(255, 107, 107, 0.3);
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Sidebar Styling */
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Enhanced Feature Cards */
        .feature-card h4 {
            color: #ffeaa7;
            margin-bottom: 15px;
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        /* Caption Header Styling */
        .caption-header {
            font-size: 1.6rem;
            font-weight: 700;
            margin: 25px 0 15px 0;
            display: flex;
            align-items: center;
            gap: 12px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }
        
        .caption-header.greedy {
            color: #ff6b6b;
        }
        
        .caption-header.beam {
            color: #4ecdc4;
        }
        
        /* Enhanced Statistics */
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: all 0.5s;
        }
        
        .stat-card:hover::before {
            left: 100%;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        
        .stat-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffeaa7;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 0.95rem;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 500;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .title-container h1 {
                font-size: 2.5rem;
            }
            
            .subtitle {
                font-size: 1.1rem;
            }
            
            .stats-container {
                grid-template-columns: 1fr;
            }
            
            .upload-container, .model-info, .feature-card {
                padding: 25px;
            }
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.5);
        }
        
        /* Enhanced metric styling */
        .metric-container {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# --- Load model and tokenizer ---
@st.cache_resource
def load_caption_model():
    model = load_model("model/caption_model.h5", compile=False)
    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# --- Configuration ---
cnn_output_dim = 2048
max_caption_length = 34

# --- Initialize session state ---
if 'captions_generated' not in st.session_state:
    st.session_state.captions_generated = 0
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = []

# --- Header ---
# --- Header ---
st.markdown("""
    <div class="title-container">
        <h1>🖼️ AI Image Caption Generator</h1>
        <p class="subtitle">Transform your images into words using advanced neural networks</p>
        <p class="subtitle">✨ Powered by Greedy & Beam Search Decoding ✨</p>
    </div>
""", unsafe_allow_html=True)

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
        <div class="upload-container">
            <h3>📄 Upload Your Image</h3>
            <p>Supported formats: JPG, PNG, JPEG</p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    st.markdown("</div>", unsafe_allow_html=True)

    # Model Information
    st.markdown("""
        <div class="model-info">
            <h3>🤖 Model Information</h3>
            <ul>
                <li><b>Architecture</b>: CNN + LSTM Encoder-Decoder</li>
                <li><b>Feature Extraction</b>: CNN Output Dimension: 2048</li>
                <li><b>Max Caption Length</b>: 34 tokens</li>
                <li><b>Decoding Methods</b>: Greedy & Beam Search</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with col2:
    if uploaded_file:
        st.markdown("""
            <div class="image-preview">
                <h3>🖼️ Image Preview</h3>
        """, unsafe_allow_html=True)

        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("✨ Generate Captions", key="generate_btn"):
            try:
                caption_model, tokenizer = load_caption_model()
                start_time = time.time()

                with st.spinner("🔄 Processing image..."):
                    st.markdown('<div class="loading-container"><div class="loading-spinner"></div></div>', unsafe_allow_html=True)

                st.info("⏳ Extracting image features...")
                image_features = preprocess_image(uploaded_file)

                st.success("✅ Features extracted! Generating captions...")
                greedy_caption = greedy_generator(image_features, tokenizer, caption_model, max_caption_length, cnn_output_dim)
                beam_caption = beam_search_generator(image_features, tokenizer, caption_model, max_caption_length, cnn_output_dim)
                end_time = time.time()
                processing_time = end_time - start_time

                st.session_state.captions_generated += 1
                st.session_state.processing_time.append(processing_time)

                st.markdown("<div class='caption-container'>", unsafe_allow_html=True)
                st.markdown("<div class='caption-header greedy'>🧠 Greedy Search Caption:</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='greedy-caption'>\"{greedy_caption}\"</div>", unsafe_allow_html=True)
                st.markdown("<div class='caption-header beam'>🚀 Beam Search Caption:</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='beam-caption'>\"{beam_caption}\"</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='stats-container'>", unsafe_allow_html=True)
                st.markdown(f'''
                    <div class="stat-card">
                        <div class="stat-value">{processing_time:.2f}s</div>
                        <div class="stat-label">Processing Time</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(greedy_caption.split())}</div>
                        <div class="stat-label">Greedy Words</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(beam_caption.split())}</div>
                        <div class="stat-label">Beam Words</div>
                    </div>
                ''', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error: {e}")

    else:
        st.markdown("""
            <div class="upload-container">
                <h3>🔹 How It Works</h3>
                <ol>
                    <li><b>Upload</b> an image using the file uploader</li>
                    <li><b>Click</b> the "Generate Captions" button</li>
                    <li><b>Wait</b> for feature extraction and processing</li>
                    <li><b>Compare</b> Greedy vs Beam Search results</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h3>🔺 Features</h3>", unsafe_allow_html=True)
    st.markdown("""
        <div class="feature-card">
            <h4>🧠 Greedy Search</h4>
            <ul>
                <li>Fast and efficient</li>
                <li>Deterministic output</li>
            </ul>
            <h4>🚀 Beam Search</h4>
            <ul>
                <li>Higher quality captions</li>
                <li>Explores multiple paths</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.captions_generated > 0:
        st.markdown("<h3>📊 Session Stats</h3>", unsafe_allow_html=True)
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.metric("Captions Generated", st.session_state.captions_generated)
        if st.session_state.processing_time:
            avg_time = sum(st.session_state.processing_time) / len(st.session_state.processing_time)
            st.metric("Avg Time", f"{avg_time:.2f}s")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h3>⚙️ Technical Details</h3>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="feature-card">
            <ul>
                <li><b>CNN Output Dim</b>: {cnn_output_dim}</li>
                <li><b>Max Caption Length</b>: {max_caption_length}</li>
                <li><b>Framework</b>: TensorFlow/Keras</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Session"):
        st.session_state.captions_generated = 0
        st.session_state.processing_time = []
        st.success("Session cleared!")

# --- Footer ---
st.markdown("""
    <hr/>
    <div style="text-align: center; color: rgba(255,255,255,0.8);">
        <p>🤖 AI Image Caption Generator | Built with TensorFlow & Streamlit</p>
        <p>✨ Compare Greedy vs Beam Search Decoding Methods ✨</p>
    </div>
""", unsafe_allow_html=True)