# Imports et Configuration Initiale
import os
import torch
import json
import logging
import uuid
import time
import cv2

from moviepy.editor import VideoFileClip
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor # Qwen 2.5 VL 3B
from transformers import pipeline # Whisper
from qwen_vl_utils import process_vision_info

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# ----- Device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Temp Folder -----
TEMP_FOLDER = os.path.join(os.getcwd(), "temp_files")
os.makedirs(TEMP_FOLDER, exist_ok=True)

# ----- Whisper (Speech-to-Text) -----
def load_whisper_model():
    WHISPER_MODEL = "openai/whisper-base"
    print(f"Chargement du modèle de transcription: {WHISPER_MODEL}...")
    try:
        transcription_pipeline = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            device=0 if torch.cuda.is_available() else -1
        )
        print("Modèle Whisper chargé.")
        return transcription_pipeline
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Whisper: {e}")
        return None

# ----- Qwen 2.5 VL 3B -----
def load_qwen_model():
    QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
    try:
        qwen_vl_processor = AutoProcessor.from_pretrained(QWEN_MODEL, trust_remote_code=True)
        qwen_vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"  # map automatiquement sur GPU ou CPU
        )
        print("Modèle Qwen chargé.")
        return qwen_vl_processor, qwen_vl_model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Qwen: {e}")
        return None, None

# --- Fonctions Utilitaires Video Preprosessing ---
def generate_unique_filename(directory, extension):
    """Génère un nom de fichier unique dans un dossier."""
    while True:
        filename = str(uuid.uuid4()) + extension
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            return filepath
        
def extract_middle_frame(temp_video_path, output_dir=TEMP_FOLDER):
    """Extrait l'image du milieu et la retourne en tant qu'objet PIL Image."""
    print(f"Extraction de l'image du milieu de : {temp_video_path}")
    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            print("Erreur: Impossible d'ouvrir la vidéo.")
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
             print("Erreur: La vidéo ne contient aucune image.")
             cap.release()
             return None
        middle_frame_index = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            print("Image du milieu extraite avec succès.")
            return image
        else:
            print("Erreur: Impossible de lire l'image du milieu.")
            return None
    except Exception as e:
        print(f"Erreur lors de l'extraction de l'image : {e}")
        return None
    
def extract_audio(temp_video_path, output_dir=TEMP_FOLDER):
    """Extrait l'audio et retourne le chemin du fichier audio temporaire."""
    print(f"Extraction de l'audio de : {temp_video_path}")
    audio_path = os.path.join(output_dir, "temp_audio.wav")
    try:
        with VideoFileClip(temp_video_path) as video_clip:
            audio_clip = video_clip.audio
            if audio_clip:
                audio_clip.write_audiofile(audio_path, codec='pcm_s16le', logger=None) # logger=None pour moins de verbosité
                audio_clip.close()
                print(f"Audio extrait et sauvegardé dans : {audio_path}")
                return audio_path
            else:
                print("Aucune piste audio trouvée dans la vidéo.")
                return None
    except Exception as e:
        print(f"Erreur lors de l'extraction de l'audio : {e}")
        if os.path.exists(audio_path): os.remove(audio_path)
        return None

def transcribe_audio(audio_path, transcription_pipeline):
    """Transcrit le fichier audio à l'aide du pipeline fourni."""
    if not transcription_pipeline:
        print("Modèle de transcription non chargé. Transcription ignorée.")
        return ""
    if not audio_path or not os.path.exists(audio_path):
        print("Chemin audio invalide ou fichier manquant. Transcription ignorée.")
        return ""
    print(f"Transcription de l'audio : {audio_path}")
    try:
        # Forcer la transcription sur le device choisi (CPU ou GPU)
        result = transcription_pipeline(audio_path)
        transcription = result["text"]
        print("Transcription terminée.")
        return transcription
    except Exception as e:
        print(f"Erreur lors de la transcription audio : {e}")
        return ""

def analyze_content(temp_image_path, transcribed_text, prompt, qwen_vl_model, qwen_vl_processor, max_new_tokens=2048):
    """Analyse l'image (PIL) et le texte avec Qwen-VL."""
    if not os.path.exists(temp_image_path):
        print("Analyse annulée : image manquante.")
        return json.dumps({"error": "Image non fournie ou invalide."})

    print("Début de l'analyse multimodale...")

    try:
        # Construction du prompt
        messages = [
            {"role": "user", "content": prompt},
            {"role": "user", "content": [
                {"type": "image", "image": temp_image_path},
                {"type": "text", "text": f"Transcription Audio: \"{transcribed_text}\""},
            ]}
        ]
    
        # Préparation
        text = qwen_vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = qwen_vl_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(qwen_vl_model.device)

        # Génération
        output_ids = qwen_vl_model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [out_ids[len(input_ids):] for input_ids, out_ids in zip(inputs.input_ids, output_ids)]
        output_text_list = qwen_vl_processor.batch_decode(generated_ids, skip_special_tokens=True)

        print("Réponse Qwen-VL reçue.")
        return output_text_list[0] if output_text_list else json.dumps({"error": "Réponse du modèle vide."})

    except Exception as e:
        print(f"Erreur lors de l'inférence : {e}")
        return json.dumps({"error": f"Erreur d'inférence : {e}"})

    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


def cleanup_files(*args):
    """Supprime les fichiers temporaires spécifiés."""
    for file_path in args:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Fichier temporaire supprimé : {file_path}")
            except Exception as e:
                print(f"Erreur lors de la suppression du fichier {file_path}: {e}")
                
print("Fonctions utilitaires définies.")   
