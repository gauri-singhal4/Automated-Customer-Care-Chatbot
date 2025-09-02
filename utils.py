"""
Utility functions for the Customer Service Chatbot
"""
import re
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def preprocess_text(self, text):
        """Complete text preprocessing pipeline"""
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned_text)
        return ' '.join(tokens)

class SentimentAnalyzer:
    """Sentiment analysis utilities"""
    
    def get_sentiment(self, text):
        """Get sentiment polarity and label"""
        if pd.isna(text) or text == "":
            return 0, 'neutral'
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return polarity, sentiment

class ModelUtils:
    """Model loading and saving utilities"""
    
    @staticmethod
    def save_model(model, filepath):
        """Save model to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        logging.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load model from file"""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logging.info(f"Model loaded from {filepath}")
            return model
        except FileNotFoundError:
            logging.error(f"Model file not found: {filepath}")
            return None
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_dataset(df, required_columns):
        """Validate dataset structure"""
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for empty dataset
        if df.empty:
            raise ValueError("Dataset is empty")
        
        return True
    
    @staticmethod
    def check_data_quality(df):
        """Check data quality metrics"""
        quality_report = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        return quality_report

class Logger:
    """Logging utilities"""
    
    @staticmethod
    def setup_logger(name, log_file, level=logging.INFO):
        """Set up logger"""
        # Create logs directory if it doesn't exist
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

def generate_timestamp():
    """Generate timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_directory(directory):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_config(config_file):
    """Load configuration from file"""
    # This can be extended to load from JSON/YAML files
    pass
