import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time # Used for exponential backoff

# --- Configuration Constants ---
MODEL_PATH = 'emotion_model\model_file.h5' # Ensure this file is in the same directory
# Haar Cascade path for face detection
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = (48, 48)

# --- Spotify Setup ---
# *** IMPORTANT: Replace with your actual Spotify Client ID and Secret ***
SPOTIFY_CLIENT_ID = '31b85c896d824248baee151898803419' 
SPOTIFY_CLIENT_SECRET = 'd9c5c9eb187e414c94ee12f50bbc82f3'

# @st.cache_resource ensures the model and detector are loaded only once
@st.cache_resource
def load_resources():
    """Loads the Keras model and Haar Cascade detector."""
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}. Check file path/existence. Error: {e}")
        st.stop()
        
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        st.warning("Could not load Haar Cascade XML. Face detection might fail. Ensure 'haarcascade_frontalface_default.xml' is accessible.")

    return model, face_cascade

# --- Spotify API Interaction with Exponential Backoff ---
def initialize_spotify():
    """Initializes and returns the Spotify client."""
    if SPOTIFY_CLIENT_ID == 'YOUR_SPOTIFY_CLIENT_ID' or SPOTIFY_CLIENT_SECRET == 'YOUR_SPOTIFY_CLIENT_SECRET':
        st.error("Please replace 'YOUR_SPOTIFY_CLIENT_ID' and 'YOUR_SPOTIFY_CLIENT_SECRET' in the code with your actual Spotify API credentials.")
        return None
        
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        ))
        return sp
    except Exception as e:
        st.error(f"Error initializing Spotify API. Check your credentials and internet connection. Error: {e}")
        return None

def get_songs(sp, emotion):
    """Fetches Spotify tracks based on the predicted emotion."""
    query_map = {
        'Happy': 'upbeat dance pop',
        'Sad': 'melancholic soothing acoustic',
        'Angry': 'calm instrumental piano',
        'Neutral': 'lofi beats chill hop',
        'Surprise': 'high energy party hits',
        'Fear': 'calm focus classical',
        'Disgust': 'peaceful ambient instrumental'
    }
    
    q = query_map.get(emotion, 'chill lofi')
    
    # Simple exponential backoff loop for API stability
    max_retries = 3
    for attempt in range(max_retries):
        try:
            results = sp.search(q=q, type='track', limit=5)
            songs = []
            for track in results['tracks']['items']:
                name = track['name']
                artist = track['artists'][0]['name']
                url = track['external_urls']['spotify']
                songs.append(f"**{name}** - {artist} [▶️ Listen on Spotify]({url})")
            return songs
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429 and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff (1s, 2s, 4s)
                st.warning(f"Spotify rate limit hit. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                st.error(f"Failed to fetch Spotify songs after {attempt+1} attempts. Error: {e}")
                return ["Could not load songs."]
        except Exception as e:
            st.error(f"An unexpected error occurred while fetching Spotify songs: {e}")
            return ["Could not load songs."]
            
    return ["Could not load songs."]

# --- Core Face Detection and Prediction ---
def process_image_and_predict(img_bytes, model, face_cascade):
    """
    Detects face, preprocesses, and predicts emotion.
    Returns: (predicted_emotion, confidence, image_with_box_as_pil)
    """
    # 1. Convert bytes to OpenCV image (BGR)
    file_bytes = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 2. Detect faces
    faces = face_cascade.detectMultiScale(
        gray_img, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return None, None, Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # For simplicity, we process the largest detected face
    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3]) 

    # 3. Pre-process the face region for the model
    # Draw bounding box on the original color image
    cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Extract, resize, and convert to 48x48 grayscale
    roi_gray = gray_img[y:y + h, x:x + w]
    cropped_face = cv2.resize(roi_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    
    # Prepare for prediction: normalize, add channel and batch dimensions (1, 48, 48, 1)
    normalized_face = cropped_face / 255.0
    reshaped_face = np.expand_dims(np.expand_dims(normalized_face, -1), 0).astype('float32')

    # 4. Predict emotion
    predictions = model.predict(reshaped_face, verbose=0)
    emotion_index = np.argmax(predictions)
    emotion = EMOTION_LABELS[emotion_index]
    confidence = predictions[0][emotion_index] * 100
    
    # Display the prediction text on the image
    text = f"{emotion} ({confidence:.1f}%)"
    cv2.putText(img_cv, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        
    # Convert the processed OpenCV image (BGR) back to RGB for Streamlit display
    result_img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    return emotion, confidence, result_img_pil

# --- Streamlit App UI ---
def main_app():
    st.set_page_config(page_title="Emotion Music Recommender", layout="centered")
    st.title("🎵 Emotion-Based Music Recommender")
    st.markdown("---")

    # Load resources
    with st.spinner('Loading Deep Learning Model and Face Detector...'):
        model, face_cascade = load_resources()

    sp = initialize_spotify()
    if sp is None:
        st.stop()
        
    st.write("### 📸 Capture Your Mood")
    st.info("Please grant camera access and take a clear, frontal photo to detect your emotion.")
    img_bytes = st.camera_input("Take a photo")

    if img_bytes:
        # Process and Predict
        with st.spinner('Analyzing your facial expression and searching Spotify...'):
            emotion, confidence, result_img_pil = process_image_and_predict(img_bytes, model, face_cascade)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Analyzed Image")
            st.image(result_img_pil, use_column_width=True)

        with col2:
            if emotion is None:
                st.error("❌ No face detected in the photo. Please try again with a clearer picture.")
            else:
                st.success(f"**Detected Emotion:** {emotion} ({confidence:.1f}%)")
                st.balloons()
                
                # Get and display songs
                st.subheader("🎧 Recommended Songs")
                songs = get_songs(sp, emotion)
                
                if songs:
                    for s in songs:
                        st.markdown(s)
                else:
                    st.warning("Could not find song suggestions. Check your Spotify API status.")

if __name__ == "__main__":
    main_app()