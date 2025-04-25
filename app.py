import streamlit as st
import os
import json
from PIL import Image
from utils import extract_middle_frame, extract_audio, transcribe_audio, analyze_content, cleanup_files
from utils import load_whisper_model, load_qwen_model 
from prompt import prompt

# Configuration de la page
st.set_page_config(page_title="🎥 AI-Powered Video Analyzer", layout="centered")

# Titre principal
st.title("🤖 🎥 AI-Powered Video Analyzer")

# Description courte
st.markdown(
    "<div style='text-align: center; font-size:18px; font-weight:500;'>Marketing Insights from User Content</div>",
    unsafe_allow_html=True
)

# Fonctionnalités principales
st.markdown("### 🔍 Main Features")
st.markdown("  🎥 **Video Content Analysis**")
st.markdown("  🧠 **Key Information Extraction**")
# 1. Upload de la vidéo
TEMP_FOLDER = os.path.join(os.getcwd(), "temp_files")
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Dictionnaire pour associer les catégories aux emojis
emoji_map = {
"number_of_people": "👤",
"people": "🧑‍🤝‍🧑",
"gender": "⚧️",
"age": "⏳",
"attire": "👔",
"recording_location": "📍",
"motivation": "💡",
"occasion": "🎉",
"viral_mood": "😊",
"relationship": "❤️",
}

def display_analysis(analysis_result_json):
    """Affiche le résultat de l'analyse avec des emojis."""
    try:
        result_dict = json.loads(analysis_result_json)
        for key, value in result_dict.items():
            emoji = emoji_map.get(key, "")
            if isinstance(value, list):
                st.subheader(f"{emoji} {key.replace('_', ' ').title()}")
                for item in value:
                    for sub_key, sub_value in item.items():
                        sub_emoji = emoji_map.get(sub_key, "")
                        st.write(f"- {sub_emoji} {sub_key.replace('_', ' ').title()}: {sub_value}")
            else:
                st.write(f"{emoji} {key.replace('_', ' ').title()}: {value}")
    except json.JSONDecodeError:
        st.error("Erreur lors de la lecture du résultat JSON.")
        st.text(analysis_result_json)

uploaded_file = st.file_uploader("📁 Upload your video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    temp_video_path = os.path.join(TEMP_FOLDER, "temp_video.mp4") # nom de la vidéo
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

# Initialiser les variables
middle_image_obj = None
temp_image_path = os.path.join(TEMP_FOLDER, "temp_frame.jpg") # nom de l'image
transcribed_text = ""
analysis_result_json = None 
audio_path = None

if st.button("Analyze"):
    with st.spinner("Video analysis in progress..."):
        try:
            st.info("🧠 Video preprocessing in progress…")

            # 2. Extraire l'image du milieu
            middle_image_obj = extract_middle_frame(temp_video_path)
            if middle_image_obj:
                st.subheader("🖼️ Frame extracted :")
                st.image(middle_image_obj, width=300)
                middle_image_obj.save(temp_image_path) # Sauvegarde temporaire pour Qwen-VL
            else:
                st.warning("⚠️ Unable to extract the frame.")
    
            # 3. Extraire l'audio
            transcription_pipeline = load_whisper_model()
            audio_path = extract_audio(temp_video_path)

            # 4. Transcrire l'audio
            if audio_path:
                transcribed_text = transcribe_audio(audio_path, transcription_pipeline)
                st.subheader("📝 Audio transcription :")
                st.info(transcribed_text)
            else:
                st.warning("🔇 No audio to transcribe.")

            # 5. Analyser l'image et le texte
            if temp_image_path:
                qwen_vl_processor, qwen_vl_model = load_qwen_model()
                print("Attributes of qwen_vl_processor :")
                print(dir(qwen_vl_processor))
                if qwen_vl_processor and qwen_vl_model:
                        analysis_result_json = analyze_content(temp_image_path, transcribed_text, prompt, qwen_vl_model, qwen_vl_processor, max_new_tokens=2048)
                        """Analyze the image and text with Qwen2.5-VL-3B."""
                        st.subheader("📊 Analysis results:")
                        display_analysis(analysis_result_json)
                else:
                        st.error("❌ Unable to proceed with the analysis without the frame.")
            else:
                 st.error("❌ The frame is missing or the path is invalid.")

        except Exception as e:
            st.error(f"🔴 An error occurred during the analysis: {e}")
        finally:
            st.info("🧹 Cleaning up temporary files...")
            cleanup_files(temp_video_path, audio_path, temp_image_path)
            st.info("✅ Cleanup completed.")
else:
    st.info("⬆️ Please upload a video to start the analysis")
