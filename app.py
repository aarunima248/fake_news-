import streamlit as st
import joblib
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# Determine base directory where this script resides
base_dir = Path(__file__).resolve().parent

# Build full file paths
vectorizer_path = base_dir / "vectorizer.jb"
model_path = base_dir / "lr_model.jb"

# Load model and vectorizer (with caching for better performance)
@st.cache_resource
def load_models():
    """Load the vectorizer and model files"""
    if not vectorizer_path.exists():
        st.error(f"❌ Vectorizer file not found!")
        st.info("Please ensure 'vectorizer.jb' is in the same directory as this script.")
        st.stop()
    
    if not model_path.exists():
        st.error(f"❌ Model file not found!")
        st.info("Please ensure 'lr_model.jb' is in the same directory as this script.")
        st.stop()
    
    try:
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

# Load models
vectorizer, model = load_models()

# App UI
st.title("🔍 Fake News Detector")
st.write("Enter a news article below to check whether it is **Fake** or **Real**.")

# Text input area
inputn = st.text_area(
    "News Article:", 
    placeholder="Paste your news article here...", 
    height=200,
    key="news_input"
)

# Check button
if st.button("Check News", type="primary", use_container_width=True):
    if inputn.strip():
        with st.spinner("Analyzing..."):
            try:
                # Transform input and make prediction
                transform_input = vectorizer.transform([inputn])
                prediction = model.predict(transform_input)
                
                # Display result with better styling
                st.markdown("---")
                if prediction[0] == 1:
                    st.success("### ✅ The News is Real!")
                    st.balloons()
                else:
                    st.error("### ❌ The News is Fake!")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Add footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.9em;'>"
    "⚠️ This is a machine learning model and may not be 100% accurate.<br>"
    "Always verify news from multiple reliable sources."
    "</p>", 
    unsafe_allow_html=True
)