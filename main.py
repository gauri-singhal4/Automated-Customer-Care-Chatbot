"""
Main script to run the Customer Service Chatbot project
"""
import argparse
import sys
from pathlib import Path
import logging
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from chatbot import CustomerServiceChatbot
from config import TRAINING_DATA, PROCESSED_DATA, MODELS_DIR
from utils import Logger

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chatbot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def generate_sample_data(n_samples=1000):
    """Generate sample training data"""
    print("Generating sample data...")
    processor = DataProcessor()
    
    # Generate sample data
    df = processor.generate_sample_data(n_samples)
    
    # Process the data
    df_processed = processor.process_dataset(df)
    
    # Save the data
    processor.save_processed_data(df_processed, TRAINING_DATA)
    
    print(f"Sample data generated and saved to {TRAINING_DATA}")
    return df_processed

def train_models():
    """Train the chatbot models"""
    print("Training models...")
    
    # Load or generate data
    processor = DataProcessor()
    df = processor.load_data(TRAINING_DATA)
    
    if df is None:
        print("No training data found. Generating sample data...")
        df = generate_sample_data()
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_pipeline(df)
    
    print("Model training completed!")
    print("Model Performance:")
    for name, result in results.items():
        print(f"  {name}: {result['accuracy']:.4f}")
    
    return results

def run_chatbot():
    """Run the chatbot in interactive mode"""
    print("Starting Customer Service Chatbot...")
    print("Type 'quit' to exit, 'help' for available commands")
    
    # Initialize chatbot
    chatbot = CustomerServiceChatbot()
    
    # Health check
    health = chatbot.health_check()
    if health['status'] != 'healthy':
        print(f"Warning: Chatbot health check failed: {health}")
        if not health['models_loaded']:
            print("Models not found. Please run 'python main.py --train' first.")
            return
    
    print("Chatbot is ready! How can I help you today?")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Thank you for using our customer service. Have a great day!")
                break
            elif user_input.lower() == 'help':
                print("\nI can help you with:")
                for intent in chatbot.get_supported_intents():
                    print(f"  • {intent.replace('_', ' ').title()}")
                continue
            elif user_input.lower() == 'health':
                health = chatbot.health_check()
                print(f"Health Status: {health}")
                continue
            elif user_input == '':
                continue
            
            # Get chatbot response
            result = chatbot.chat(user_input)
            
            print(f"\nBot: {result['response']}")
            
            # Show debug info if confidence is low
            if result['confidence'] < 0.6:
                print(f"(Debug: Intent={result['intent']}, Confidence={result['confidence']:.3f})")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def test_chatbot():
    """Test the chatbot with predefined queries"""
    print("Testing chatbot...")
    
    chatbot = CustomerServiceChatbot()
    
    test_queries = [
        "I need to check my account balance",
        "My card is lost, please block it",
        "I want to pay my loan EMI",
        "I have a complaint about poor service",
        "My bill payment failed",
        "Hello, how are you?",  # Test fallback
        ""  # Test empty input
    ]
    
    results = chatbot.batch_process(test_queries)
    
    for result in results:
        print(f"\nQuery: {result['query']}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Response: {result['response']}")
        print(f"Needs Escalation: {result['needs_escalation']}")
        print("-" * 50)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Customer Service Chatbot")
    parser.add_argument('--generate-data', action='store_true', 
                       help='Generate sample training data')
    parser.add_argument('--train', action='store_true', 
                       help='Train the chatbot models')
    parser.add_argument('--chat', action='store_true', 
                       help='Run interactive chatbot')
    parser.add_argument('--test', action='store_true', 
                       help='Test chatbot with predefined queries')
    parser.add_argument('--web', action='store_true', 
                       help='Start web interface')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples to generate (default: 1000)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Create necessary directories
    MODELS_DIR.mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    try:
        if args.generate_data:
            generate_sample_data(args.samples)
        elif args.train:
            train_models()
        elif args.test:
            test_chatbot()
        elif args.web:
            print("Starting web interface...")
            import app
            app.run_app()
        elif args.chat:
            run_chatbot()
        else:
            # Default: run interactive chatbot
            print("No specific command provided. Running interactive chatbot...")
            print("Use --help to see available options")
            run_chatbot()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
