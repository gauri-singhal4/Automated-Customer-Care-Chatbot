import streamlit as st
import random

st.title("🤖 Customer Service Chatbot - Quick Demo")

# Simple responses
responses = {
    'balance': "I'll help you check your account balance right away.",
    'card': "I can help you block your card immediately for security.",
    'loan': "I'll assist you with your loan payment inquiry.", 
    'complaint': "I'll make sure your complaint gets proper attention.",
    'bill': "I can help you with your bill payment issue."
}

user_input = st.text_input("Ask me anything about banking:")

if user_input:
    # Simple keyword detection
    if 'balance' in user_input.lower():
        st.success(responses['balance'])
    elif any(word in user_input.lower() for word in ['card', 'block', 'lost']):
        st.success(responses['card'])
    elif 'loan' in user_input.lower():
        st.success(responses['loan'])
    elif 'complaint' in user_input.lower():
        st.success(responses['complaint'])  
    elif 'bill' in user_input.lower():
        st.success(responses['bill'])
    else:
        st.info("I can help with balance, cards, loans, complaints, or bill payments!")

st.markdown("**Demo Features:** Intent detection, Response generation, Web interface")