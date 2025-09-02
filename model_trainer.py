"""
Model training module for the Customer Service Chatbot
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import logging
from pathlib import Path
import joblib
from utils import ModelUtils
from config import MODEL_CONFIG, MODELS_DIR

class ModelTrainer:
    """Handle model training operations"""
    
    def __init__(self):
        self.vectorizer = None
        self.models = {}
        self.best_model = None
        self.label_encoder = LabelEncoder()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Features and target
        X = df['processed_text']
        y = df['intent']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state'],
            stratify=y
        )
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(
            max_features=MODEL_CONFIG['max_features'],
            ngram_range=MODEL_CONFIG['ngram_range']
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        self.logger.info(f"Data prepared: Train shape {X_train_tfidf.shape}, Test shape {X_test_tfidf.shape}")
        
        return X_train_tfidf, X_test_tfidf, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare performance"""
        
        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(
                random_state=MODEL_CONFIG['random_state'],
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                random_state=MODEL_CONFIG['random_state'],
                n_estimators=100
            )
        }
        
        results = {}
        
        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'classification_report': classification_rep
            }
            
            self.logger.info(f"{name} Accuracy: {accuracy:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        self.best_model = results[best_model_name]['model']
        self.models = results
        
        self.logger.info(f"Best model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
        
        return results
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        evaluation = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return evaluation
    
    def save_models(self):
        """Save trained models"""
        if not MODELS_DIR.exists():
            MODELS_DIR.mkdir(parents=True)
        
        # Save best model
        if self.best_model:
            model_path = MODELS_DIR / "best_intent_classifier.pkl"
            ModelUtils.save_model(self.best_model, model_path)
        
        # Save vectorizer
        if self.vectorizer:
            vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
            ModelUtils.save_model(self.vectorizer, vectorizer_path)
        
        # Save all models
        all_models_path = MODELS_DIR / "all_models.pkl"
        ModelUtils.save_model(self.models, all_models_path)
        
        self.logger.info("Models saved successfully")
    
    def load_models(self):
        """Load trained models"""
        # Load best model
        model_path = MODELS_DIR / "best_intent_classifier.pkl"
        self.best_model = ModelUtils.load_model(model_path)
        
        # Load vectorizer
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        self.vectorizer = ModelUtils.load_model(vectorizer_path)
        
        return self.best_model is not None and self.vectorizer is not None
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance for the best model"""
        if not self.best_model or not self.vectorizer:
            return None
        
        # Only works for models with feature_importances_ attribute
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = self.vectorizer.get_feature_names_out()
            importances = self.best_model.feature_importances_
            
            # Get top features
            top_indices = np.argsort(importances)[-top_n:]
            top_features = [(feature_names[i], importances[i]) for i in top_indices]
            
            return sorted(top_features, key=lambda x: x[1], reverse=True)
        
        return None
    
    def predict_intent(self, text):
        """Predict intent for given text"""
        if not self.best_model or not self.vectorizer:
            raise ValueError("Models not loaded. Please load models first.")
        
        # Preprocess text (this should be done by the preprocessor)
        text_tfidf = self.vectorizer.transform([text])
        
        # Predict
        predicted_intent = self.best_model.predict(text_tfidf)[0]
        confidence = max(self.best_model.predict_proba(text_tfidf)[0])
        
        return predicted_intent, confidence
    
    def train_pipeline(self, df):
        """Complete training pipeline"""
        self.logger.info("Starting training pipeline...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Train models
        results = self.train_models(X_train, X_test, y_train, y_test)
        
        # Save models
        self.save_models()
        
        self.logger.info("Training pipeline completed")
        
        return results
