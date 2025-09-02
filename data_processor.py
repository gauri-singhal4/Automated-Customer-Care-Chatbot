"""
Data processing module for the Customer Service Chatbot
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from utils import TextPreprocessor, SentimentAnalyzer, DataValidator
from config import MODEL_CONFIG, INTENT_CATEGORIES

class DataProcessor:
    """Handle data processing operations"""
    
    def __init__(self):
        self.text_processor = TextPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.validator = DataValidator()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_sample_data(self, n_samples=1000):
        """Generate sample dataset for training"""
        np.random.seed(42)
        
        # Sample complaint narratives for different intents
        complaint_narratives = {
            'balance_inquiry': [
                "I need to check my account balance but cannot access online banking",
                "Unable to see my current balance on mobile app",
                "Need balance information for my savings account",
                "Can you help me check my account balance",
                "I want to know my current account balance"
            ],
            'card_management': [
                "My card is lost and I need to block it immediately",
                "I want to unblock my card that was blocked yesterday",
                "Need to report my stolen credit card",
                "My debit card is not working, please help",
                "I need to activate my new credit card"
            ],
            'loan_assistance': [
                "I need help with my loan EMI payment",
                "Want to know my outstanding loan amount",
                "Need assistance with loan prepayment",
                "My loan payment is overdue, what should I do",
                "I want to apply for a personal loan"
            ],
            'complaint_management': [
                "I am not satisfied with the previous resolution",
                "Want to file a complaint about poor service",
                "My issue was not resolved properly",
                "I had a bad experience at the branch",
                "The customer service was very poor"
            ],
            'bill_payments': [
                "Unable to pay my credit card bill online",
                "Need reminder for upcoming bill payment",
                "Payment failed but amount was debited",
                "I want to set up automatic bill payments",
                "My bill payment is not reflecting in the system"
            ]
        }
        
        # Generate data
        data = []
        date_received = pd.date_range(start='2023-01-01', end='2024-12-31', periods=n_samples)
        
        products = ['Credit Card', 'Debit Card', 'Personal Loan', 'Home Loan', 'Savings Account', 'Current Account']
        sub_products = ['Visa', 'MasterCard', 'Standard', 'Premium', 'Basic', 'Gold']
        issues = ['Balance Inquiry', 'Transaction Dispute', 'Card Block', 'Card Unblock', 'Lost Card', 'Loan Payment', 'Bill Payment', 'Account Access']
        sub_issues = ['Online Issue', 'ATM Issue', 'Mobile App', 'Branch Visit', 'Phone Call']
        
        for i in range(n_samples):
            intent = np.random.choice(INTENT_CATEGORIES)
            narrative = np.random.choice(complaint_narratives[intent])
            
            row = {
                'Date_received': date_received[i],
                'Product': np.random.choice(products),
                'Sub_product': np.random.choice(sub_products),
                'Issue': np.random.choice(issues),
                'Sub_issue': np.random.choice(sub_issues),
                'Consumer_complaint_narrative': narrative,
                'Company_public_response': np.random.choice(['Yes', 'No']),
                'Company': f'Bank_{np.random.randint(1, 6)}',
                'State': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL']),
                'ZIP_code': np.random.randint(10000, 99999),
                'Tags': np.random.choice(['Older American', 'Military', 'Student', 'None']),
                'Consumer_consent_provided': np.random.choice(['Yes', 'No']),
                'Submitted_via': np.random.choice(['Web', 'Phone', 'Email', 'Mobile App']),
                'Date_sent_to_company': date_received[i] + timedelta(days=np.random.randint(1, 3)),
                'Company_response_to_consumer': np.random.choice(['Closed with explanation', 'Closed', 'In progress']),
                'Timely_response': np.random.choice(['Yes', 'No']),
                'Consumer_disputed': np.random.choice(['Yes', 'No']),
                'Complaint_ID': f'COMP_{i+1:06d}',
                'intent': intent,
                'query': narrative,
                'response': f"Thank you for contacting us regarding {intent.replace('_', ' ')}. We will assist you with this matter.",
                'Priority_Level': np.random.choice(['High', 'Medium', 'Low']),
                'Resolution_Status': np.random.choice(['Resolved', 'Pending', 'Escalated']),
                'Customer_Satisfaction_Score': np.random.randint(1, 6),
                'Agent_ID': f'AGT_{np.random.randint(1, 51):03d}',
                'Resolution_Time_Hours': np.random.randint(1, 73),
                'Category': intent.replace('_', ' ').title()
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        self.logger.info(f"Generated sample dataset with {len(df)} samples")
        return df
    
    def process_dataset(self, df):
        """Process the dataset for machine learning"""
        # Validate dataset
        required_columns = ['Consumer_complaint_narrative', 'intent']
        self.validator.validate_dataset(df, required_columns)
        
        # Process text
        self.logger.info("Processing text data...")
        df['processed_text'] = df['Consumer_complaint_narrative'].apply(
            self.text_processor.preprocess_text
        )
        
        # Add text features
        df['text_length'] = df['Consumer_complaint_narrative'].str.len()
        df['word_count'] = df['Consumer_complaint_narrative'].str.split().str.len()
        
        # Add sentiment analysis
        self.logger.info("Performing sentiment analysis...")
        sentiment_results = df['Consumer_complaint_narrative'].apply(
            self.sentiment_analyzer.get_sentiment
        )
        df['sentiment_score'] = [result[0] for result in sentiment_results]
        df['sentiment'] = [result[1] for result in sentiment_results]
        
        # Convert date columns
        if 'Date_received' in df.columns:
            df['Date_received'] = pd.to_datetime(df['Date_received'])
            df['Year'] = df['Date_received'].dt.year
            df['Month'] = df['Date_received'].dt.month
            df['Day_of_Week'] = df['Date_received'].dt.day_name()
            df['Quarter'] = df['Date_received'].dt.quarter
        
        self.logger.info("Dataset processing completed")
        return df
    
    def save_processed_data(self, df, filepath):
        """Save processed dataset"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=False)
        self.logger.info(f"Processed data saved to {filepath}")
    
    def load_data(self, filepath):
        """Load dataset from file"""
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Data loaded from {filepath}")
            return df
        except FileNotFoundError:
            self.logger.error(f"Data file not found: {filepath}")
            return None
    
    def get_data_summary(self, df):
        """Get summary statistics of the dataset"""
        summary = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'intent_distribution': df['intent'].value_counts().to_dict() if 'intent' in df.columns else {},
            'data_types': df.dtypes.to_dict()
        }
        return summary
