import streamlit as st
import joblib
import numpy as np
from deep_translator import GoogleTranslator
from gtts import gTTS
import uuid
import os
import requests

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="🩺 Symptom2Disease Multilingual Predictor", layout="centered")
st.title("🧠 Symptom2Disease: ML Diagnosis Assistant")
st.warning("⚠️ This AI tool provides educational suggestions based on symptoms. It is NOT a substitute for professional medical advice.")

# ------------------- LOAD API KEY & MODEL -------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
model = joblib.load("disease_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ------------------- DATA -------------------
disease_labels = [
    '(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne', 'Alcoholic hepatitis',
    'Allergy', 'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis',
    'Chicken pox', 'Chronic cholestasis', 'Common Cold', 'Dengue', 'Diabetes',
    'Dimorphic hemmorhoids(piles)', 'Drug Reaction', 'Fungal infection', 'GERD',
    'Gastroenteritis', 'Heart attack', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D',
    'Hepatitis E', 'Hypertension', 'Hyperthyroidism', 'Hypoglycemia',
    'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine',
    'Osteoarthristis', 'Paralysis (brain hemorrhage)', 'Peptic ulcer diseae',
    'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid', 'Urinary tract infection',
    'Varicose veins', 'hepatitis A'
]

all_symptoms = sorted([
    "fever", "cough", "headache", "rash", "vomiting", "diarrhea", "fatigue",
    "chest pain", "abdominal pain", "nausea", "sore throat", "breathlessness",
    "joint pain", "muscle pain", "dizziness", "sweating", "weight loss", "itching"
])

# ------------------- SIDEBAR -------------------
st.sidebar.title("🔧 Settings")
language = st.sidebar.selectbox("🌍 Select your language:", ["English", "Urdu", "Punjabi", "Spanish", "French", "Arabic"])
lang_code_map = {"English": "en", "Urdu": "ur", "Punjabi": "pa", "Spanish": "es", "French": "fr", "Arabic": "ar"}
lang_code = lang_code_map[language]

# ------------------- SYMPTOM INPUT -------------------
st.subheader("📝 Select or Type Your Symptoms")
selected_symptoms = st.multiselect("Choose from common symptoms:", options=all_symptoms)
typed_symptoms = st.text_input("Or enter additional symptoms (comma-separated):")
user_input = ", ".join(selected_symptoms + [s.strip() for s in typed_symptoms.split(",") if s.strip()])

# ------------------- SESSION STATE TO STORE DISEASES -------------------
if "predicted" not in st.session_state:
    st.session_state.predicted = []

# ------------------- PREDICT BUTTON -------------------
if st.button("🔍 Predict Diseases"):
    if not user_input.strip():
        st.warning("⚠️ Please select or type at least one symptom.")
    else:
        try:
            translated_input = GoogleTranslator(source='auto', target='en').translate(user_input)
        except:
            st.error("🌐 Translation failed. Please try again.")
            st.stop()

        cleaned_symptoms = ', '.join([s.strip() for s in translated_input.split(',') if s.strip()])
        X_input = vectorizer.transform([cleaned_symptoms])
        preds = model.predict_proba(X_input)[0]
        top_indices = np.argsort(preds)[::-1][:3]

        st.session_state.predicted = [(disease_labels[i], preds[i]) for i in top_indices]

        st.markdown("### ✅ Top 3 Predicted Diseases (Confidence Only):")
        for i, (disease, prob) in enumerate(st.session_state.predicted):
            badge = "🟢 High" if prob >= 0.75 else "🟡 Medium" if prob >= 0.5 else "🔴 Low"
            st.markdown(f"**{i+1}. {disease}** — {badge} ({prob:.2f} confidence)")

# ------------------- EXPLAIN BUTTON -------------------
if st.session_state.predicted and st.button("🧾 Explain Diseases"):
    full_text = ""
    st.markdown("### 🧠 Explanation of Predicted Diseases:")

    for i, (disease, prob) in enumerate(st.session_state.predicted):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-8b-8192",
                    "messages": [
                        {"role": "system", "content": "You are a medical assistant."},
                        {"role": "user", "content": f"Explain the disease {disease} in simple terms."}
                    ]
                }
            )
            explanation = response.json()["choices"][0]["message"]["content"]
        except:
            explanation = f"{disease} is predicted with {prob:.2f} confidence."

        if language != "English":
            try:
                explanation = GoogleTranslator(source='en', target=lang_code).translate(explanation)
            except:
                st.warning(f"🌐 Could not translate explanation for {disease}.")

        st.markdown(f"**{i+1}. {disease}** — {explanation}")
        full_text += f"{i+1}. {disease}: {explanation}\n"

    # ------------------- TTS IN SIDEBAR -------------------
    try:
        tts_lang = lang_code if lang_code in ['en', 'ur', 'es', 'fr', 'ar', 'pa'] else 'en'
        tts = gTTS(text=full_text, lang=tts_lang)
        file_path = f"audio_{uuid.uuid4().hex}.mp3"
        tts.save(file_path)
        with open(file_path, "rb") as audio_file:
            with st.sidebar:
                st.subheader("🎧 Listen to Explanations")
                st.audio(audio_file.read(), format="audio/mp3")
        os.remove(file_path)
    except:
        with st.sidebar:
            st.warning("🔇 Audio generation failed.")
