🎵 EmoTune: Real-Time Emotion-Based Music Recommender

Data Set Link===> https://www.kaggle.com/datasets/msambare/fer2013

🌟 Project Tagline

EmoTune is a cutting-edge web application that leverages Deep Learning (CNNs) and computer vision to analyze a user's real-time facial expression and instantly generate a personalized Spotify playlist matching their detected mood.

💡 The Problem Solved

How often do you struggle to find the right music for your current mood? EmoTune eliminates the guesswork. This application provides a seamless, intuitive bridge between your emotional state and the perfect soundtrack, moving beyond manual input to deliver genuine, context-aware song recommendations.

✨ Key Features

Real-Time Emotion Detection: Utilizes a pre-trained Convolutional Neural Network (CNN) model (model_file.h5) to classify live facial expressions into 7 categories (Happy, Sad, Angry, Neutral, etc.).

Accurate Face Cropping: Employs OpenCV's Haar Cascade to precisely detect and crop the face from the camera input, ensuring the deep learning model receives the optimal input for high-accuracy prediction.

Personalized Spotify Playlists: Maps the detected emotion directly to curated search queries (e.g., 'Angry' -> 'calm instrumental piano') using the Spotify Web API to retrieve relevant tracks.

Interactive Web Interface: Built using Streamlit for a fast, beautiful, and user-friendly experience, requiring only a single photo capture to receive results.

Robust API Handling: Implements Exponential Backoff to manage rate limiting and ensure stable communication with the Spotify API.

🛠️ Tech Stack
<img width="975" height="456" alt="image" src="https://github.com/user-attachments/assets/d78e52fc-f47f-4ad9-96b2-38453b7ced90" />

🚀 Getting Started

Follow these steps to set up and run EmoTune locally.

1. Prerequisites

Python 3.8+

A Spotify Developer Account for API credentials (Client ID and Secret).

Open the music_recommender.py file and replace the placeholder credentials:

# music_recommender.py snippet
SPOTIFY_CLIENT_ID = 'YOUR_SPOTIFY_CLIENT_ID' 
SPOTIFY_CLIENT_SECRET = 'YOUR_SPOTIFY_CLIENT_SECRET'

Add the Model

Ensure your trained Keras model, named model_file.h5, is placed in the root directory of the project

Run the Application

Execute the Streamlit script from your terminal:

streamlit run app.py

The application will open automatically in your browser. Grant camera access, and you're ready to get your mood-based soundtrack!

🔮 Future Enhancements

Continuous Prediction: Implement live video streaming for continuous emotion tracking and dynamic playlist updates.

User Feedback Loop: Allow users to rate recommendations to fine-tune the emotion-to-query mapping.

Spotify Authorization: Switch from Client Credentials Flow to Authorization Code Flow to recommend songs directly from the user's saved playlists.

Advanced Face Landmarks: Use models like Mediapipe for more precise landmark-based emotion analysis.

🤝 Contribution

Contributions are welcome! If you have suggestions or want to improve a feature, please feel free to submit a Pull Request or open an issue.
