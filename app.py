import streamlit as st
import os
import json
from PIL import Image
from utils import extract_middle_frame, extract_audio, transcribe_audio, analyze_content, cleanup_files
from utils import load_whisper_model, load_qwen_model 
from prompt import prompt

st.set_page_config(page_title="🎥 AI Content Analyzer", layout="centered")
st.title("🤖 🎥 AI-Powered Content Analyzer")

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

uploaded_file = st.file_uploader("📁 Upload your video to analyze", type=["mp4", "mov", "avi", "mkv"])

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

if st.button("Lancer l'analyse"):
    with st.spinner("Analyse de la vidéo en cours..."):
        try:
            st.info("🧠 Prétraitement de la vidéo en cours...")

            # 2. Extraire l'image du milieu
            middle_image_obj = extract_middle_frame(temp_video_path)
            if middle_image_obj:
                st.subheader("🖼️ Image extraite du milieu:")
                st.image(middle_image_obj, use_column_width=True)
                middle_image_obj.save(temp_image_path) # Sauvegarde temporaire pour Qwen-VL
            else:
                st.warning("⚠️ Impossible d'extraire l'image du milieu.")
    
            # 3. Extraire l'audio
            transcription_pipeline = load_whisper_model()
            audio_path = extract_audio(temp_video_path)

            # 4. Transcrire l'audio
            if audio_path:
                transcribed_text = transcribe_audio(audio_path, transcription_pipeline)
                st.subheader("📝 Transcription audio")
                st.info(transcribed_text)
            else:
                st.warning("🔇 Aucun audio à transcrire.")

            # 5. Analyser l'image et le texte
            if temp_image_path:
                qwen_vl_processor, qwen_vl_model = load_qwen_model()
                print("Attributs de qwen_vl_processor :")
                print(dir(qwen_vl_processor))
                if qwen_vl_processor and qwen_vl_model:
                        analysis_result_json = analyze_content(temp_image_path, transcribed_text, prompt, qwen_vl_model, qwen_vl_processor, max_new_tokens=2048)
                        """Analyse l'image (PIL) et le texte avec Qwen-VL."""
                        st.subheader("📊 Résultat de l'analyse:")
                        display_analysis(analysis_result_json)
                else:
                        st.error("❌ Impossible de procéder à l'analyse sans l'image du milieu.")
            else:
                 st.error("❌ L'image du milieu est introuvable ou le chemin est invalide.")

        except Exception as e:
            st.error(f"🔴 Une erreur s'est produite pendant l'analyse : {e}")
        finally:
            st.info("🧹 Nettoyage des fichiers temporaires...")
            cleanup_files(temp_video_path, audio_path, temp_image_path)
            st.info("✅ Nettoyage terminé.")
else:
    st.info("⬆️ Veuillez télécharger une vidéo pour lancer l'analyse.")
