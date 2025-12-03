import streamlit as st
import pandas as pd
import numpy as np
import nltk
from joblib import load
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

# --- NLTK Resource Setup ---
# Check if NLTK resources are downloaded, if not, try to download them.
# In a Streamlit environment, this helps ensure they are available.
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.info("Downloading NLTK resources...")
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('wordnet')
        nltk.download('stopwords')
        st.success("NLTK Resources downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {e}")
        st.stop() # Stop the script if critical resources can't be downloaded

# Initialize NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Preprocessing Function (Copied from your training script) ---
def preprocess_text(text):
    """
    Standard preprocessing pipeline for text:
    1. Lowercase
    2. Remove punctuation/special chars
    3. Tokenization
    4. Stopword Removal
    5. Lemmatization
    """
    if not isinstance(text, str):
        return "" # Handle non-string input safely

    # 1. Convert to Lowercase
    text = text.lower()
    
    # 2. Remove Punctuation and Special Characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Stopword Removal
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoin tokens into a single string
    return ' '.join(tokens)

# --- Chatbot Core Function ---
# Predefined fixed responses (Retrieval System)
RESPONSES = {
    "ask_room_price": "Our rooms start from RM180 per night.",
    "ask_availability": "We currently have several rooms available.",
    "ask_facilities": "We offer free Wi-Fi, breakfast, pool, gym and parking.",
    "ask_location": "We are located in Kuala Lumpur City Centre (KLCC).",
    "ask_checkin_time" : "Check-in time is from 2:00 PM.",
    "ask_checkout_time" : "Check-out time is at 12:00 PM.",
    "ask_booking" : "You can book directly through our website or at the front desk.",
    "ask_cancellation" : "Cancellations are free up to 24 hours before arrival.",
    "greeting" : "Hello! How may I assist you today?",
    "goodbye" : "Goodbye! Have a great day!"
}

def chatbot_reply_nb(user_input, model, vectorizer, responses):
    # 1. Preprocessing (using the full function for consistency)
    cleaned_input = preprocess_text(user_input)

    # 2. Feature Extraction: Transform the cleaned input using the fitted vectorizer
    if cleaned_input:
        vector = vectorizer.transform([cleaned_input])
    else:
        return "I couldn't process your input. Please try a different question.", "unknown_error"

    # 3. Intent Prediction
    intent = model.predict(vector)[0]

    # 4. Retrieval 
    response = responses.get(intent, f"Sorry, I predicted the intent '{intent}', but I don't have a specific response for that yet. Please rephrase your question.")
    
    return response, intent

# --- Streamlit Application Layout and Logic ---

# Title and Description
st.title("🛎️ Hotel Booking Intent Chatbot")
st.markdown("""
This chatbot uses a **Multinomial Naive Bayes** model trained on TF-IDF features 
to classify user intent and provide a relevant, predefined response.
""")

# Load Model and Vectorizer (Use st.cache_resource for efficiency)
@st.cache_resource
def load_resources():
    model_path = 'naive_bayes_intent_model.joblib'
    vectorizer_path = 'tfidf_vectorizer.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error(f"Required files not found: '{model_path}' and/or '{vectorizer_path}'.")
        st.warning("Please ensure you run your training script first to save the model and vectorizer.")
        st.stop()
        
    try:
        model = load(model_path)
        vectorizer = load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        st.stop()

nb_model, vectorizer = load_resources()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the hotel (e.g., 'What is the check-in time?'):"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate chatbot response
    with st.chat_message("assistant"):
        # Get the response and the predicted intent
        response, predicted_intent = chatbot_reply_nb(prompt, nb_model, vectorizer, RESPONSES)
        
        # Display the main response
        st.markdown(response)
        
        # Display the predicted intent as a debug/info note
        st.caption(f"🤖 Predicted Intent: **{predicted_intent}**")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Optional: Display a list of known intents for guidance
st.sidebar.header("Known Intents")
st.sidebar.info("The chatbot can answer questions related to these topics:")
st.sidebar.text(f"- {', '.join(RESPONSES.keys())}")