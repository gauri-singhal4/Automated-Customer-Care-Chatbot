import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Customer Service Chatbot", page_icon="🤖")

st.title("🤖 Customer Service Chatbot")
st.markdown("**AI-Powered Customer Support Assistant**")

# Simple chatbot interface
st.subheader("💬 Chat with our AI Assistant")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.text_input("How can I help you today?", key="user_input")

if user_input:
    # Simple intent detection
    response = ""
    intent = ""
    
    if any(word in user_input.lower() for word in ['balance', 'account', 'money']):
        response = "I'll help you check your account balance. Please provide your account details."
        intent = "Balance Inquiry"
    elif any(word in user_input.lower() for word in ['card', 'block', 'lost']):
        response = "I can help you block your card immediately for security. Is your card lost or stolen?"
        intent = "Card Management"
    elif any(word in user_input.lower() for word in ['loan', 'emi', 'payment']):
        response = "I'll assist you with your loan payment inquiry. What specific help do you need?"
        intent = "Loan Assistance"
    elif any(word in user_input.lower() for word in ['complaint', 'problem', 'issue']):
        response = "I apologize for the inconvenience. I'll escalate your complaint for proper resolution."
        intent = "Complaint Management"
    elif any(word in user_input.lower() for word in ['bill', 'pay', 'due']):
        response = "I can help you with your bill payment. What payment issue are you facing?"
        intent = "Bill Payment"
    else:
        response = "I can help with balance inquiries, card issues, loans, complaints, or bill payments. How can I assist you?"
        intent = "General"
    
    # Add to chat history
    st.session_state.chat_history.append((user_input, response, intent))
    
    # Display response
    st.success(f"**Bot:** {response}")
    st.info(f"**Detected Intent:** {intent}")

# Display chat history
if st.session_state.chat_history:
    st.subheader("📋 Chat History")
    for i, (user_msg, bot_msg, detected_intent) in enumerate(st.session_state.chat_history, 1):
        with st.expander(f"Conversation {i}"):
            st.write(f"**You:** {user_msg}")
            st.write(f"**Bot:** {bot_msg}")
            st.write(f"**Intent:** {detected_intent}")

# Quick action buttons
st.subheader("🚀 Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💰 Check Balance"):
        st.success("I'll help you check your account balance right away.")

with col2:
    if st.button("🔒 Block Card"):
        st.success("I can help you block your card immediately for security.")

with col3:
    if st.button("🏠 Loan Help"):
        st.success("I'll assist you with your loan payment inquiry.")

# Project info
st.markdown("---")
st.markdown("**🤖 Customer Service Chatbot** | Built with Streamlit | Ready for Production")
