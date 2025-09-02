"""
Configuration file for the Customer Service Chatbot project
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Model files
MODEL_FILE = MODELS_DIR / "best_intent_classifier.pkl"
VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.pkl"
RESPONSE_TEMPLATES_FILE = MODELS_DIR / "response_templates.pkl"
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder.pkl"

# Dataset files
TRAINING_DATA = DATA_DIR / "customer_service_dataset.csv"
PROCESSED_DATA = DATA_DIR / "processed_dataset.csv"

# Model configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "max_features": 1000,
    "ngram_range": (1, 2),
    "confidence_threshold": {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4
    }
}

# Intent categories
INTENT_CATEGORIES = [
    'balance_inquiry',
    'card_management',
    'loan_assistance',
    'complaint_management',
    'bill_payments'
]

# Response templates
RESPONSE_TEMPLATES = {
    'balance_inquiry': [
        "I'll help you check your account balance. Let me retrieve that information for you.",
        "To assist you with your balance inquiry, I'll need to verify your account details.",
        "I can help you check your current balance. Please provide your account information."
    ],
    'card_management': [
        "I understand you need help with your card. I'll assist you with blocking/unblocking or reporting it.",
        "For card-related issues, I can help you immediately block your card for security.",
        "Let me help you with your card management request right away."
    ],
    'loan_assistance': [
        "I'll assist you with your loan-related query. Let me check your loan details.",
        "For loan assistance, I can help you with payments, balance inquiries, or general information.",
        "I'm here to help with your loan concerns. Let me gather the necessary information."
    ],
    'complaint_management': [
        "I apologize for any inconvenience. I'll escalate your complaint to ensure proper resolution.",
        "Your complaint is important to us. I'll make sure it's properly documented and addressed.",
        "I'll help you file this complaint and ensure it receives appropriate attention."
    ],
    'bill_payments': [
        "I'll assist you with your bill payment issue. Let me check the payment status.",
        "For bill payment problems, I can help you with alternative payment methods or troubleshooting.",
        "I understand your payment concern. Let me help resolve this issue quickly."
    ]
}

# Flask configuration
FLASK_CONFIG = {
    "DEBUG": True,
    "HOST": "0.0.0.0",
    "PORT": 5000,
    "SECRET_KEY": "your-secret-key-here"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": BASE_DIR / "logs" / "chatbot.log"
}

