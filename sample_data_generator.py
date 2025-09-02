"""
Sample data generator for the Customer Service Chatbot
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_comprehensive_sample_data(n_samples=5000):
    """Create a comprehensive sample dataset"""
    np.random.seed(42)
    
    # Extended complaint narratives for better training
    complaint_narratives = {
        'balance_inquiry': [
            "I need to check my account balance but cannot access online banking",
            "Unable to see my current balance on mobile app",
            "Need balance information for my savings account",
            "Can you help me check my account balance",
            "I want to know my current account balance",
            "Please provide my account balance details",
            "How much money do I have in my account",
            "I need to verify my account balance",
            "Can you tell me my available balance",
            "I want to check my checking account balance"
        ],
        'card_management': [
            "My card is lost and I need to block it immediately",
            "I want to unblock my card that was blocked yesterday",
            "Need to report my stolen credit card",
            "My debit card is not working, please help",
            "I need to activate my new credit card",
            "Please block my card as it's been stolen",
            "I lost my wallet, need to secure my cards",
            "My card was damaged, need a replacement",
            "I want to temporarily block my card for security",
            "Need help with card activation process"
        ],
        'loan_assistance': [
            "I need help with my loan EMI payment",
            "Want to know my outstanding loan amount",
            "Need assistance with loan prepayment",
            "My loan payment is overdue, what should I do",
            "I want to apply for a personal loan",
            "Can you help me with loan restructuring",
            "I need information about my home loan",
            "What's my remaining loan tenure",
            "I want to make partial loan prepayment",
            "Need help calculating my loan interest"
        ],
        'complaint_management': [
            "I am not satisfied with the previous resolution",
            "Want to file a complaint about poor service",
            "My issue was not resolved properly",
            "I had a bad experience at the branch",
            "The customer service was very poor",
            "I want to escalate my previous complaint",
            "Not happy with the service quality",
            "I faced discrimination at your branch",
            "Want to file a formal complaint",
            "The staff was very rude to me"
        ],
        'bill_payments': [
            "Unable to pay my credit card bill online",
            "Need reminder for upcoming bill payment",
            "Payment failed but amount was debited",
            "I want to set up automatic bill payments",
            "My bill payment is not reflecting in the system",
            "Having trouble with online bill payment",
            "Can you help me pay my utility bills",
            "Need assistance with mobile recharge",
            "My payment was successful but not updated",
            "Want to schedule recurring payments"
        ]
    }
    
    # Generate more realistic data
    data = []
    date_received = pd.date_range(start='2022-01-01', end='2024-12-31', periods=n_samples)
    
    products = ['Credit Card', 'Debit Card', 'Personal Loan', 'Home Loan', 'Car Loan', 'Savings Account', 'Current Account', 'Fixed Deposit']
    sub_products = ['Visa', 'MasterCard', 'Rupay', 'Standard', 'Premium', 'Basic', 'Gold', 'Platinum']
    issues = ['Balance Inquiry', 'Transaction Dispute', 'Card Block', 'Card Unblock', 'Lost Card', 'Loan Payment', 'Bill Payment', 'Account Access', 'Interest Rate', 'Fees']
    sub_issues = ['Online Issue', 'ATM Issue', 'Mobile App', 'Branch Visit', 'Phone Call', 'Email', 'Chat']
    companies = ['ABC Bank', 'XYZ Financial', 'Digital Bank Corp', 'National Bank', 'Regional Credit Union']
    states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    
    intent_categories = list(complaint_narratives.keys())
    
    for i in range(n_samples):
        intent = np.random.choice(intent_categories)
        narrative = np.random.choice(complaint_narratives[intent])
        
        # Add some variation to narratives
        if np.random.random() < 0.3:  # 30% chance to add context
            context_phrases = [
                "This is urgent.",
                "Please help me quickly.",
                "I've been waiting for days.",
                "This is the third time I'm calling.",
                "I'm very frustrated."
            ]
            narrative += " " + np.random.choice(context_phrases)
        
        row = {
            # Original columns
            'Date_received': date_received[i],
            'Product': np.random.choice(products),
            'Sub_product': np.random.choice(sub_products),
            'Issue': np.random.choice(issues),
            'Sub_issue': np.random.choice(sub_issues),
            'Consumer_complaint_narrative': narrative,
            'Company_public_response': np.random.choice(['Yes', 'No'], p=[0.7, 0.3]),
            'Company': np.random.choice(companies),
            'State': np.random.choice(states),
            'ZIP_code': np.random.randint(10000, 99999),
            'Tags': np.random.choice(['Older American', 'Military', 'Student', 'None'], p=[0.15, 0.1, 0.2, 0.55]),
            'Consumer_consent_provided': np.random.choice(['Yes', 'No'], p=[0.8, 0.2]),
            'Submitted_via': np.random.choice(['Web', 'Phone', 'Email', 'Mobile App'], p=[0.4, 0.3, 0.15, 0.15]),
            'Date_sent_to_company': date_received[i] + timedelta(days=np.random.randint(0, 3)),
            'Company_response_to_consumer': np.random.choice(['Closed with explanation', 'Closed', 'In progress'], p=[0.6, 0.3, 0.1]),
            'Timely_response': np.random.choice(['Yes', 'No'], p=[0.85, 0.15]),
            'Consumer_disputed': np.random.choice(['Yes', 'No'], p=[0.15, 0.85]),
            'Complaint_ID': f'COMP_{i+1:06d}',
            'intent': intent,
            'query': narrative,
            'response': f"Thank you for contacting us regarding {intent.replace('_', ' ')}. We will assist you with this matter promptly.",
            
            # Additional recommended columns
            'Priority_Level': np.random.choice(['High', 'Medium', 'Low'], p=[0.2, 0.5, 0.3]),
            'Resolution_Status': np.random.choice(['Resolved', 'Pending', 'Escalated'], p=[0.7, 0.2, 0.1]),
            'Customer_Satisfaction_Score': np.random.randint(1, 6),
            'Agent_ID': f'AGT_{np.random.randint(1, 101):03d}',
            'Resolution_Time_Hours': np.random.exponential(24),  # More realistic distribution
            'Category': intent.replace('_', ' ').title(),
            'Channel_Rating': np.random.randint(1, 6),
            'Follow_up_Required': np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
        }
        
        data.append(row)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate sample data
    df = create_comprehensive_sample_data(5000)
    
    # Save to CSV
    output_path = Path(__file__).parent / "customer_service_dataset.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Sample data generated: {len(df)} records")
    print(f"Saved to: {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Intent distribution:\n{df['intent'].value_counts()}")
