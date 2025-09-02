"""
Main chatbot class for the Customer Service Chatbot
"""
import numpy as np
import logging
from pathlib import Path
from utils import TextPreprocessor, ModelUtils
from model_trainer import ModelTrainer
from config import RESPONSE_TEMPLATES, MODEL_CONFIG, MODELS_DIR

class CustomerServiceChatbot:
    """Main chatbot class"""
    
    def __init__(self):
        self.text_processor = TextPreprocessor()
        self.model_trainer = ModelTrainer()
        self.response_templates = RESPONSE_TEMPLATES
        self.confidence_thresholds = MODEL_CONFIG['confidence_threshold']
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load models if available
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            success = self.model_trainer.load_models()
            if success:
                self.logger.info("Models loaded successfully")
                return True
            else:
                self.logger.warning("Failed to load models")
                return False
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def predict_intent(self, user_input):
        """Predict intent from user input"""
        try:
            # Preprocess the input
            processed_input = self.text_processor.preprocess_text(user_input)
            
            # Predict intent
            intent, confidence = self.model_trainer.predict_intent(processed_input)
            
            return intent, confidence
        except Exception as e:
            self.logger.error(f"Error predicting intent: {e}")
            return None, 0.0
    
    def generate_response(self, intent, confidence, user_input=""):
        """Generate response based on intent and confidence"""
        if intent is None:
            return "I'm sorry, I couldn't understand your request. Could you please rephrase it?"
        
        # Get appropriate response template
        if intent in self.response_templates:
            templates = self.response_templates[intent]
            base_response = np.random.choice(templates)
        else:
            base_response = "I'll help you with your request."
        
        # Modify response based on confidence
        if confidence >= self.confidence_thresholds['high']:
            return f"{base_response}"
        elif confidence >= self.confidence_thresholds['medium']:
            return f"{base_response} Could you provide more details to better assist you?"
        elif confidence >= self.confidence_thresholds['low']:
            return f"I think you're asking about {intent.replace('_', ' ')}. {base_response} Please let me know if I understood correctly."
        else:
            return ("I'm not entirely sure about your request. Could you please rephrase or provide more details? "
                   "I'll connect you with a human agent if needed.")
    
    def get_escalation_response(self):
        """Get escalation response for complex queries"""
        return ("I understand this might be a complex issue. I'm transferring you to one of our human agents "
                "who can provide you with more detailed assistance. Please hold on.")
    
    def get_fallback_response(self):
        """Get fallback response when intent cannot be determined"""
        return ("I'm sorry, I didn't quite understand that. Here are some things I can help you with:\n"
                "• Check your account balance\n"
                "• Block or unblock your card\n"
                "• Assist with loan payments\n"
                "• Help with bill payments\n"
                "• Handle complaints or concerns\n\n"
                "Please let me know which one you need help with, or describe your issue in more detail.")
    
    def chat(self, user_input):
        """Main chat function"""
        if not user_input or user_input.strip() == "":
            return "Please enter your query."
        
        # Predict intent
        intent, confidence = self.predict_intent(user_input)
        
        # Log the interaction
        self.logger.info(f"User input: {user_input}")
        self.logger.info(f"Predicted intent: {intent}, Confidence: {confidence:.3f}")
        
        # Generate response
        if intent is None or confidence < self.confidence_thresholds['low']:
            response = self.get_fallback_response()
        else:
            response = self.generate_response(intent, confidence, user_input)
        
        return {
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'needs_escalation': confidence < self.confidence_thresholds['low']
        }
    
    def batch_process(self, queries):
        """Process multiple queries at once"""
        results = []
        for query in queries:
            result = self.chat(query)
            results.append({
                'query': query,
                **result
            })
        return results
    
    def get_conversation_context(self):
        """Get conversation context (for future implementation)"""
        # This can be extended to maintain conversation context
        return {}
    
    def set_response_templates(self, new_templates):
        """Update response templates"""
        self.response_templates.update(new_templates)
        self.logger.info("Response templates updated")
    
    def get_supported_intents(self):
        """Get list of supported intents"""
        return list(self.response_templates.keys())
    
    def health_check(self):
        """Check if chatbot is working properly"""
        try:
            test_result = self.chat("Hello")
            return {
                'status': 'healthy',
                'models_loaded': self.model_trainer.best_model is not None,
                'vectorizer_loaded': self.model_trainer.vectorizer is not None,
                'test_response': test_result['response']
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'models_loaded': False,
                'vectorizer_loaded': False
            }
