import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration Parameters ---
MAX_SEQUENCE_LENGTH = 20  # Max number of words used during training
CONFIDENCE_THRESHOLD = 0.75 # New: Threshold to classify as "unrecognized intent"

# --- A. CHATBOT RESPONSE LOOKUP TABLE ---
# This dictionary maps the predicted intent name (string) to a fixed response.
RESPONSE_DICT = {
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

# --- B. Model Loading and Caching ---
# Use Streamlit's caching mechanism to load resources only once
@st.cache_resource
def load_resources():
    """Loads the model, tokenizer, and label encoder from files."""
    try:
        # Load Keras Model
        model = tf.keras.models.load_model('lstm_intent_model.h5')
        
        # Load Tokenizer
        tokenizer = joblib.load('tokenizerLSTM.joblib')
        
        # Load LabelEncoder
        le = joblib.load('label_encoder.joblib')
        
        return model, tokenizer, le
    except FileNotFoundError as e:
        st.error(f"Error loading required model files. Please ensure all files (lstm_intent_model.h5, tokenizer.joblibLSTM, label_encoder.joblib) are in the same directory. Missing file: {e.filename}")
        return None, None, None

model, tokenizer, le = load_resources()

# --- C. Prediction Function ---
def predict_intent(text):
    """
    Predicts the intent of a given text and returns the response.
    Includes logic for 'unrecognized intent' based on confidence.
    """
    if model is None or tokenizer is None or le is None:
        return "Model not loaded. Check the file paths."

    # 1. Preprocess the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, 
                                    maxlen=MAX_SEQUENCE_LENGTH, 
                                    padding='post', 
                                    truncating='post')

    # 2. Make Prediction
    predictions = model.predict(padded_sequence, verbose=0)
    
    # Get the index (ID) of the highest probability
    predicted_id = np.argmax(predictions, axis=1)[0]
    # Get the confidence score (the highest probability)
    confidence_score = np.max(predictions, axis=1)[0]
    
    # 3. Apply Confidence Threshold Logic
    if confidence_score < CONFIDENCE_THRESHOLD:
        # Intent is considered 'unrecognized'
        intent_name = "unrecognized_intent"
        # The corresponding response for 'unrecognized_intent' is retrieved from RESPONSE_DICT
        response = RESPONSE_DICT.get(intent_name)
    else:
        # Convert the predicted ID back to the intent name
        intent_name = le.inverse_transform([predicted_id])[0]
        # Retrieve the specific response for the predicted intent
        response = RESPONSE_DICT.get(intent_name, RESPONSE_DICT['unrecognized_intent'])

    # Format the confidence score to percentage
    confidence_display = f"{confidence_score*100:.2f}%"
    
    return intent_name, response, confidence_display


# --- D. Streamlit App Layout (替换后的聊天记录模式) ---
def main():
    st.set_page_config(page_title="Hotel Intent Chatbot", layout="centered")

    st.title("🛎️ Hotel Intent Recognition Chatbot")
    st.caption(f"LSTM Model | Confidence Threshold: **{CONFIDENCE_THRESHOLD*100:.0f}%**")

    # 1. 初始化聊天历史 (Session State)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # 增加一个初始的问候消息
        st.session_state.messages.append({"role": "assistant", "content": RESPONSE_DICT['greeting']})

    # 2. 显示聊天历史
    # 使用 st.chat_message 来渲染对话气泡
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # 如果是助手的回复，额外显示置信度和意图
            if message["role"] == "assistant" and "intent" in message:
                st.caption(f"Intent: **{message['intent']}** | Confidence: **{message['confidence']}**")
            st.markdown(message["content"])

    # 3. 处理用户输入
    # 使用 st.chat_input 替换 st.text_input 和 st.button
    user_input = st.chat_input("How may I assist you with your booking?")
    
    if user_input:
        # 3a. 将用户输入添加到历史记录并显示
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 立即在界面上显示用户输入
        with st.chat_message("user"):
            st.markdown(user_input)

        # 3b. 进行预测并生成回复
        with st.spinner('Analyzing query...'):
            intent_name, response, confidence_display = predict_intent(user_input)
            
            # 3c. 将助手回复添加到历史记录
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "intent": intent_name,
                "confidence": confidence_display
            })

            # 3d. 在界面上显示助手回复
            with st.chat_message("assistant"):
                # 高亮显示意图和置信度
                st.caption(f"Intent: **{intent_name}** | Confidence: **{confidence_display}**")
                st.markdown(response)

if __name__ == "__main__":
    main()
