import streamlit as st
import requests

st.set_page_config(page_title="MindSupport - AI Mental Health Chatbot", page_icon="🧠", layout="centered")

st.title("MindSupport: AI-Driven Mental Health Chatbot")

user_input = st.text_input("How are you feeling today?", max_chars=200)

if st.button("Submit"):
    if user_input:
        try:
            response = requests.post("http://localhost:5000/chatbot", json={"input": user_input})
            if response.status_code == 200:
                result = response.json()
                st.write(f"Chatbot: {result['response']}")
                st.write(f"Sentiment Detected: {result['sentiment']}")
            else:
                st.error("Error connecting to the chatbot service.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.write("In case of crisis, please contact the following helpline: 1-800-123-4567")
