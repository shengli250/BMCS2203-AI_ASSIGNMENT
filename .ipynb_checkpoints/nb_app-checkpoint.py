import streamlit as st
import joblib
import json
import pandas as pd
import os

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="Hotel Chatbot (Naive Bayes)",
    page_icon="🏨",
    layout="wide"
)

# --- 2. 加载资源的函数 (使用缓存提高速度) ---

@st.cache_resource
def load_model_and_vectorizer():
    """加载模型和向量化器"""
    try:
        model = joblib.load('naive_bayes_intent_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizerNB.joblib')
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"❌ 找不到模型文件: {e}")
        return None, None

@st.cache_data
def load_responses():
    """加载 JSON 回复库"""
    try:
        with open('response.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ 找不到 response.json 文件")
        return {}
    except json.JSONDecodeError:
        st.error("❌ response.json 文件格式错误")
        return {}

@st.cache_data
def load_dataset():
    """加载 CSV 数据集用于预览"""
    try:
        return pd.read_csv('dataset.csv')
    except Exception:
        return None

# --- 3. 初始化加载 ---
model, vectorizer = load_model_and_vectorizer()
responses = load_responses()
df = load_dataset()

# --- 4. 预测逻辑函数 ---
def get_prediction(text):
    if not model or not vectorizer:
        return "System Error", "模型未加载", 0.0
    
    # 预处理
    text_clean = text.lower()
    
    # 向量化
    vector = vectorizer.transform([text_clean])
    
    # 预测意图
    intent = model.predict(vector)[0]
    
    # 获取置信度 (Probability) - 选做，用于展示模型有多确信
    probs = model.predict_proba(vector)[0]
    max_prob = max(probs)
    
    # 获取回复
    reply = responses.get(intent, "Sorry, I'm not sure how to answer that.")
    
    return intent, reply, max_prob

# --- 5. 侧边栏 (Sidebar) ---
with st.sidebar:
    st.header("🤖 模型控制台")
    st.write("这是一个基于 Naive Bayes 的意图识别聊天机器人。")
    
    st.divider()
    
    # 显示模型状态
    if model and vectorizer:
        st.success("✅ 模型已加载")
    else:
        st.error("❌ 模型加载失败")

    if responses:
        st.success(f"✅ 已加载 {len(responses)} 条回复规则")

    # 数据集预览
    st.divider()
    st.subheader("📊 训练数据预览")
    if df is not None:
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"共 {len(df)} 条数据")
    else:
        st.warning("未找到 dataset.csv")

# --- 6. 主聊天界面 ---
st.title("🏨 Hotel Assistant Bot")
st.caption("Ask me about room prices, check-in times, or facilities!")

# 初始化聊天历史 (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your hotel booking today?"}]

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 7. 处理用户输入 ---
if prompt := st.chat_input("Type your message here..."):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 获取模型预测
    intent, reply, confidence = get_prediction(prompt)

    # 3. 显示机器人回复
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
        
        # 可选：在回复下方显示调试信息 (意图和置信度)
        with st.expander("🔍 Debug Info (Model Prediction)"):
            st.write(f"**Predicted Intent:** `{intent}`")
            st.write(f"**Confidence:** `{confidence:.2%}`")