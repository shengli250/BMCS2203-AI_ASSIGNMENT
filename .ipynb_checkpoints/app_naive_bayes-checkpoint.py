import streamlit as st
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# --- Configuration Parameters ---
CONFIDENCE_THRESHOLD = 0.75 # 置信度阈值：低于此值则视为“无法识别的意图”

# --- A. CHATBOT RESPONSE LOOKUP TABLE ---
RESPONSE_DICT = {
    # 使用您训练代码中的响应字典，但调整为更正式的 Streamlit 部署格式
    "ask_room_price": "Our rooms start from RM180 per night.",
    "ask_availability": "We currently have several rooms available.",
    "ask_facilities": "We offer free Wi-Fi, breakfast, pool, gym and parking.",
    "ask_location": "We are located in Kuala Lumpur City Centre (KLCC).",
    "ask_checkin_time" : "Check-in time is from 2:00 PM.",
    "ask_checkout_time" : "Check-out time is at 12:00 PM.",
    "ask_booking" : "You can book directly through our website or at the front desk.",
    "ask_cancellation" : "Cancellations are free up to 24 hours before arrival.",
    "greeting" : "Hello! How may I assist you today?",
    "goodbye" : "Goodbye! Have a great day!",
    # Default response for unrecognized intents
    "unrecognized_intent": "I apologize, but I currently cannot understand your request. Could you please try rephrasing your question?",
}

# --- B. NLTK Download and Preprocessing Setup ---
# 使用 st.cache_resource 来确保 NLTK 资源只下载一次
@st.cache_resource(show_spinner="Downloading NLTK resources...")
def download_nltk_resources():
    """Downloads necessary NLTK resources into the Streamlit cache."""
    try:
        # NLTK 下载
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True) # 这通常不是必需的
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        # 初始化 NLTK 对象
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        return True, stop_words, lemmatizer
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {e}")
        return False, set(), None

is_nltk_ready, stop_words, lemmatizer = download_nltk_resources()

def preprocess_text(text):
    """Applies the same preprocessing steps as the training script."""
    if not lemmatizer:
        return "" # Handle case where NLTK setup failed
        
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

# --- C. Model Loading and Caching ---
@st.cache_resource
def load_resources():
    """Loads the model and vectorizer from files."""
    try:
        # Load MultinomialNB Model
        nb_model = joblib.load('naive_bayes_intent_model.joblib')
        
        # Load TFIDF Vectorizer
        vectorizer = joblib.load('tfidf_vectorizerNB.joblib')
        
        return nb_model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Error loading required model files. Please ensure all files (naive_bayes_intent_model.joblib, tfidf_vectorizer.joblib) are in the same directory. Missing file: {e.filename}")
        return None, None

nb_model, vectorizer = load_resources()

# --- D. Prediction Function ---
def predict_intent(text):
    """
    Predicts the intent using the Naive Bayes model and applies a confidence threshold.
    """
    if nb_model is None or vectorizer is None or not is_nltk_ready:
        return "setup_error", RESPONSE_DICT.get("unrecognized_intent"), "N/A"

    # 1. Preprocessing and Feature Extraction
    user_input_cleaned = preprocess_text(text)
    vector = vectorizer.transform([user_input_cleaned])

    # 2. Get Probability Predictions
    # MultinomialNB provides probabilities via predict_proba
    predictions_proba = nb_model.predict_proba(vector)[0]
    
    # Get the index (ID) of the highest probability
    predicted_index = np.argmax(predictions_proba)
    # Get the confidence score (the highest probability)
    confidence_score = np.max(predictions_proba)
    
    # Get the predicted intent name (MultinomialNB.classes_ contains the intent names)
    predicted_intent_name = nb_model.classes_[predicted_index]

    # 3. Apply Confidence Threshold Logic
    if confidence_score < CONFIDENCE_THRESHOLD:
        intent_name = "unrecognized_intent"
        response = RESPONSE_DICT.get(intent_name)
    else:
        intent_name = predicted_intent_name
        # Retrieve the specific response for the predicted intent
        response = RESPONSE_DICT.get(intent_name, RESPONSE_DICT['unrecognized_intent'])

    confidence_display = f"{confidence_score*100:.2f}%"
    
    return intent_name, response, confidence_display


# --- E. Streamlit App Layout ---
def main():
    st.set_page_config(page_title="NB Intent Chatbot (TF-IDF)", layout="centered")

    st.title("🤖 Hotel Intent Recognition Chatbot (Naive Bayes)")
    st.markdown("基于 **TF-IDF** 和 **Multinomial Naive Bayes** 模型")

    st.info(f"**Confidence Threshold for Unrecognized Intent:** {CONFIDENCE_THRESHOLD*100:.0f}% (用于判断模型置信度是否足够)")
    
    # User Input
    user_input = st.text_input("**Your Query:**", placeholder="E.g., I want to book a room. Do you have a gym?")
    
    # Create the button ONLY ONCE
    button_clicked = st.button("🚀 **Get Chatbot Response**") 

    if button_clicked:
        if user_input:
            with st.spinner('Analyzing query...'):
                # Predict the intent
                intent_name, response, confidence_display = predict_intent(user_input)
                
                # --- Display Results ---
                st.markdown("---")
                
                st.subheader("💡 Analysis Result")
                
                # Highlight the predicted intent
                if intent_name == "unrecognized_intent" or intent_name == "setup_error":
                    st.error(f"**Predicted Intent:** `{intent_name}` (Confidence: {confidence_display})")
                else:
                    st.success(f"**Predicted Intent:** `{intent_name}` (Confidence: {confidence_display})")

                st.subheader("💬 Chatbot Response")
                st.markdown(f"> **{response}**")
                
        else:
            # Handle the case where the button is clicked but the input is empty
            st.warning("Please enter a query to get a response.")

if __name__ == "__main__":
    main()