import streamlit as st
import joblib
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import hashlib

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector - Advanced",
    page_icon="🔍",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
    }
    
    .stApp, .stMarkdown, p, span, label {
        color: #ffffff !important;
    }
    
    .stTextArea textarea {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #ffffff !important;
        opacity: 0.6;
    }
    
    .stButton button {
        background-color: #ff4b4b !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 500 !important;
    }
    
    .stButton button:hover {
        background-color: #ff6b6b !important;
    }
    
    [data-testid="stSuccess"] {
        background-color: #1a4d2e !important;
    }
    
    [data-testid="stSuccess"] h3 {
        color: #4ade80 !important;
    }
    
    [data-testid="stError"] {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%) !important;
        border: 1px solid #ef4444 !important;
        box-shadow: 0 4px 6px -1px rgba(220, 38, 38, 0.3) !important;
    }
    
    [data-testid="stError"] h3 {
        color: #fecaca !important;
    }
    
    [data-testid="stInfo"] {
        background-color: #1a2a4d !important;
    }
    
    [data-testid="stWarning"] {
        background-color: #4d3d1a !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    hr {
        border-color: #333333 !important;
    }
    
    .metric-card {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333333;
        margin: 10px 0;
    }
    
    .data-source-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.85em;
        margin: 5px;
        font-weight: 500;
    }
    
    .twitter-badge {
        background-color: #1DA1F2;
        color: white;
    }
    
    .facebook-badge {
        background-color: #4267B2;
        color: white;
    }
    
    .whatsapp-badge {
        background-color: #25D366;
        color: white;
    }
    
    .news-badge {
        background-color: #FF6B35;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for data collection
if 'collected_data' not in st.session_state:
    st.session_state.collected_data = []

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Determine base directory
base_dir = Path(__file__).resolve().parent
vectorizer_path = base_dir / "vectorizer.jb"
model_path = base_dir / "lr_model.jb"

# Fake news correction database
FAKE_NEWS_CORRECTIONS = {
    "council staff told how to address fake news online": {
        "correct_news": "South Tyneside Council conducted a training session on identifying and responding to misinformation, not on hiding or correcting information. The session was about media literacy and helping staff recognize false information to better serve the public."
    },
    "staff at south tyneside council are being advised when to hide or correct misinformation": {
        "correct_news": "South Tyneside Council conducted a training session on identifying and responding to misinformation, not on hiding or correcting information. The session was about media literacy and helping staff recognize false information to better serve the public."
    },
    "vaccine causes autism": {
        "correct_news": "Extensive scientific research has conclusively proven that vaccines do not cause autism. The original study claiming this link was fraudulent and has been retracted. Vaccines are safe and effective."
    },
    "5g causes coronavirus": {
        "correct_news": "There is no scientific evidence linking 5G technology to COVID-19. Viruses cannot travel on radio waves or mobile networks. COVID-19 is caused by the SARS-CoV-2 virus and spreads through respiratory droplets."
    }
}

def find_correction(news_text):
    """Search for known fake news patterns and return correction if found"""
    normalized_text = news_text.lower().strip()
    
    for fake_pattern, correction_data in FAKE_NEWS_CORRECTIONS.items():
        if fake_pattern in normalized_text:
            return correction_data["correct_news"]
    
    return None

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

def generate_content_id(content):
    """Generate unique ID for content"""
    return hashlib.md5(content.encode()).hexdigest()[:12]

def collect_data(content, source, metadata, prediction, confidence=None):
    """Collect data for multimodal analysis"""
    data_entry = {
        "content_id": generate_content_id(content),
        "content": content,
        "source": source,
        "timestamp": datetime.now().isoformat(),
        "prediction": "Real" if prediction == 1 else "Fake",
        "confidence": confidence,
        "metadata": metadata
    }
    st.session_state.collected_data.append(data_entry)
    st.session_state.analysis_history.append(data_entry)
    return data_entry

def save_collected_data():
    """Save collected data to JSON file"""
    if st.session_state.collected_data:
        filename = f"collected_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(st.session_state.collected_data, f, indent=2)
        return filename
    return None

# Load models
vectorizer, model = load_models()

# App UI
st.title("🔍 Advanced Fake News Detection System")
st.write("Multi-modal data collection and analysis platform")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📰 Detection", "📊 Data Collection", "💾 Export"])

# TAB 1: Detection
with tab1:
    st.header("News Verification")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Data source selection
        data_source = st.selectbox(
            "Select Data Source:",
            ["News Article", "Twitter/X", "Facebook", "WhatsApp", "Other Social Media"],
            key="source_select"
        )
        
        # Text input
        inputn = st.text_area(
            "Content to Analyze:", 
            placeholder="Paste news article, tweet, or social media post here...", 
            height=200,
            key="news_input"
        )
    
    with col2:
        st.subheader("Metadata")
        
        # Metadata collection
        author = st.text_input("Author/Username:", placeholder="Optional", key="author")
        url = st.text_input("Source URL:", placeholder="Optional", key="url")
        shared_by = st.text_input("Shared By:", placeholder="Optional", key="shared_by")
        share_count = st.number_input("Share Count:", min_value=0, value=0, key="shares")
        
        # Additional metadata
        include_metadata = st.checkbox("Include timestamp metadata", value=True)
    
    # Check button
    if st.button("🔍 Analyze Content", type="primary", use_container_width=True):
        if inputn.strip():
            with st.spinner("Analyzing content..."):
                try:
                    # Transform input and make prediction
                    transform_input = vectorizer.transform([inputn])
                    prediction = model.predict(transform_input)
                    
                    # Get prediction probability if available
                    try:
                        proba = model.predict_proba(transform_input)
                        confidence = max(proba[0]) * 100
                    except:
                        confidence = None
                    
                    # Collect metadata
                    metadata = {
                        "author": author if author else "Unknown",
                        "url": url if url else "Not provided",
                        "shared_by": shared_by if shared_by else "Unknown",
                        "share_count": share_count,
                        "timestamp": datetime.now().isoformat() if include_metadata else None,
                        "content_length": len(inputn),
                        "word_count": len(inputn.split())
                    }
                    
                    # Collect data
                    data_entry = collect_data(inputn, data_source, metadata, prediction[0], confidence)
                    
                    # Display result
                    st.markdown("---")
                    
                    result_col1, result_col2 = st.columns([2, 1])
                    
                    with result_col1:
                        if prediction[0] == 1:
                            st.success("### ✅ The Content is Real!")
                            st.balloons()
                        else:
                            st.error("### ❌ The Content is Fake!")
                            
                            # Check for correction
                            correction = find_correction(inputn)
                            
                            if correction:
                                st.markdown("---")
                                st.info("### 📰 Correct Information:")
                                st.markdown(
                                    f"""
                                    <div style='background-color: #1a1a1a; padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3;'>
                                        <p style='color: #ffffff; margin: 0;'>{correction}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    
                    with result_col2:
                        st.markdown("### 📊 Analysis Details")
                        if confidence:
                            st.metric("Confidence", f"{confidence:.1f}%")
                        st.metric("Source", data_source)
                        st.metric("Words", metadata['word_count'])
                        
                        if share_count > 0:
                            st.metric("Shares", share_count)
                    
                    # Display metadata
                    with st.expander("📋 View Full Metadata"):
                        st.json(metadata)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

# TAB 2: Data Collection
with tab2:
    st.header("📊 Multimodal Data Collection")
    
    st.write("Collected data from multiple sources for comprehensive analysis")
    
    if st.session_state.collected_data:
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_items = len(st.session_state.collected_data)
        fake_count = sum(1 for item in st.session_state.collected_data if item['prediction'] == 'Fake')
        real_count = total_items - fake_count
        
        sources = [item['source'] for item in st.session_state.collected_data]
        unique_sources = len(set(sources))
        
        with col1:
            st.markdown("""
                <div class='metric-card'>
                    <h3>📊 Total Items</h3>
                    <h2>{}</h2>
                </div>
            """.format(total_items), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='metric-card'>
                    <h3>✅ Real News</h3>
                    <h2>{}</h2>
                </div>
            """.format(real_count), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='metric-card'>
                    <h3>❌ Fake News</h3>
                    <h2>{}</h2>
                </div>
            """.format(fake_count), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
                <div class='metric-card'>
                    <h3>🌐 Data Sources</h3>
                    <h2>{}</h2>
                </div>
            """.format(unique_sources), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display collected data
        st.subheader("Recent Collections")
        
        for idx, item in enumerate(reversed(st.session_state.collected_data[-10:])):
            with st.expander(f"📄 {item['source']} - {item['prediction']} ({item['timestamp'][:19]})"):
                col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.write("**Content Preview:**")
                    st.write(item['content'][:300] + "..." if len(item['content']) > 300 else item['content'])
                
                with col_b:
                    st.write("**Metadata:**")
                    st.write(f"- Source: {item['source']}")
                    st.write(f"- Prediction: {item['prediction']}")
                    if item['confidence']:
                        st.write(f"- Confidence: {item['confidence']:.1f}%")
                    st.write(f"- Author: {item['metadata']['author']}")
                    st.write(f"- Shares: {item['metadata']['share_count']}")
    else:
        st.info("No data collected yet. Start analyzing content in the Detection tab!")

# TAB 3: Export
with tab3:
    st.header("💾 Export Data")
    
    if st.session_state.collected_data:
        st.write(f"You have **{len(st.session_state.collected_data)}** items collected.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export to JSON", use_container_width=True):
                filename = save_collected_data()
                if filename:
                    st.success(f"✅ Data exported to {filename}")
                    
                    # Provide download button
                    with open(filename, 'r') as f:
                        st.download_button(
                            label="⬇️ Download JSON File",
                            data=f.read(),
                            file_name=filename,
                            mime="application/json"
                        )
        
        with col2:
            if st.button("📊 Export to CSV", use_container_width=True):
                df = pd.DataFrame(st.session_state.collected_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="⬇️ Download CSV File",
                    data=csv,
                    file_name=f"collected_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.success("✅ CSV ready for download")
        
        st.markdown("---")
        
        # Preview data
        st.subheader("📋 Data Preview")
        df_preview = pd.DataFrame(st.session_state.collected_data)
        st.dataframe(df_preview[['content_id', 'source', 'prediction', 'timestamp']], use_container_width=True)
        
        # Clear data option
        st.markdown("---")
        if st.button("🗑️ Clear All Collected Data", type="secondary"):
            st.session_state.collected_data = []
            st.session_state.analysis_history = []
            st.rerun()
    else:
        st.info("No data to export yet. Start collecting data in the Detection tab!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888888; font-size: 0.9em;'>"
    "⚠️ This is a machine learning model and may not be 100% accurate.<br>"
    "Always verify news from multiple reliable sources.<br>"
    "🔒 All data is stored locally in this session."
    "</p>", 
    unsafe_allow_html=True
)