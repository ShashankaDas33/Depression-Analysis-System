import streamlit as st
import os
from main import analyze_depression
import tempfile
import tensorflow as tf

st.set_page_config(
    page_title="Depression Analysis System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Depression Analysis System")
st.markdown("""
This application analyzes video content to detect potential signs of depression through:
- Facial emotion analysis
- Audio feature analysis
- Speech content analysis
""")

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name

    # Add a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Update status
        status_text.text("Analyzing video...")
        progress_bar.progress(20)

        # Run the analysis
        with st.spinner('Processing video... This may take a few minutes.'):
            # Create a container for results
            results_container = st.container()
            
            # Run analysis
            analyze_depression(video_path)
            
            # Update progress
            progress_bar.progress(100)
            status_text.text("Analysis complete!")

            # Display results in a nice format
            with results_container:
                st.success("Analysis Results")
                
                # Read and display the transcription
                if os.path.exists("transcription.txt"):
                    with open("transcription.txt", "r", encoding="utf-8") as f:
                        transcription = f.read()
                        st.subheader("Transcription")
                        st.text_area("Speech Content", transcription, height=150)

                # Display depression scores
                st.subheader("Depression Likelihood Scores")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Facial Emotion Score", f"{depression_score_face:.2%}")
                with col2:
                    st.metric("Audio Feature Score", f"{depression_score_audio:.2%}")
                with col3:
                    st.metric("Text Analysis Score", f"{depression_score_text:.2%}")

                # Display final score
                st.subheader("Final Depression Likelihood")
                st.metric("Overall Score", f"{final_score}%")

                # Display matched words if available
                if 'matched_words' in locals():
                    st.subheader("Detected Depression-related Words")
                    st.write(", ".join(matched_words))

    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(video_path):
            os.unlink(video_path)

else:
    st.info("Please upload a video file to begin analysis.")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Note: This tool is for educational purposes only and should not be used as a substitute for professional medical advice.</p>
</div>
""", unsafe_allow_html=True)

print(tf.__version__) 