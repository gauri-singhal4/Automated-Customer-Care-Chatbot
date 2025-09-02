"""
Streamlit web application for the Customer Service Chatbot
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from chatbot import CustomerServiceChatbot
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from config import INTENT_CATEGORIES

# Configure page
st.set_page_config(
    page_title="Customer Service Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.stAlert {
    margin-top: 1rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.user-message {
    background-color: #E3F2FD;
    margin-left: 20%;
}
.bot-message {
    background-color: #F5F5F5;
    margin-right: 20%;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = CustomerServiceChatbot()
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'conversation_count' not in st.session_state:
    st.session_state.conversation_count = 0

# Sidebar
with st.sidebar:
    st.title("🤖 Customer Service Bot")
    
    # Navigation
    page = st.selectbox(
        "Choose a page:",
        ["Chat Interface", "Analytics Dashboard", "Model Training", "System Health"]
    )
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("Quick Stats")
    st.metric("Conversations", st.session_state.conversation_count)
    st.metric("Supported Intents", len(INTENT_CATEGORIES))
    
    # Health check
    health = st.session_state.chatbot.health_check()
    if health['status'] == 'healthy':
        st.success("✅ System Healthy")
    else:
        st.error("❌ System Issues")

# Main content based on selected page
if page == "Chat Interface":
    st.title("💬 Customer Service Chat")
    
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Chat with our AI Assistant")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (user_msg, bot_response, metadata) in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {user_msg}
                </div>
                """, unsafe_allow_html=True)
                
                # Bot message
                confidence_text = f" (Confidence: {metadata.get('confidence', 0):.1%})" if 'confidence' in metadata else ""
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>Bot:</strong> {bot_response}{confidence_text}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Type your message:", placeholder="How can I help you today?")
            col_submit, col_clear = st.columns


# Add this to your existing Jupyter notebook cell 11, but for Streamlit integration:

# In streamlit_app.py, add this function and page
def show_wordcloud_analysis():
    st.title("☁️ Word Cloud Analysis")
    
    # Load processed data
    @st.cache_data
    def load_processed_data():
        processor = DataProcessor()
        df = processor.generate_sample_data(1000)
        return processor.process_dataset(df)
    
    df = load_processed_data()
    
    # Intent selection
    selected_intent = st.selectbox("Select Intent for Word Cloud:", 
                                  options=['All'] + list(df['intent'].unique()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate word cloud
        if selected_intent == 'All':
            text_data = ' '.join(df['processed_text'].astype(str))
            st.subheader("Word Cloud - All Intents")
        else:
            filtered_df = df[df['intent'] == selected_intent]
            text_data = ' '.join(filtered_df['processed_text'].astype(str))
            st.subheader(f"Word Cloud - {selected_intent.title()}")
        
        if text_data.strip():
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(text_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    with col2:
        # Top words analysis
        st.subheader("Top Keywords")
        
        if selected_intent == 'All':
            word_freq = pd.Series(' '.join(df['processed_text']).split()).value_counts().head(20)
        else:
            filtered_df = df[df['intent'] == selected_intent]
            word_freq = pd.Series(' '.join(filtered_df['processed_text']).split()).value_counts().head(20)
        
        fig = px.bar(
            x=word_freq.values,
            y=word_freq.index,
            orientation='h',
            title="Top 20 Keywords"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment analysis by intent
    st.subheader("Sentiment Analysis by Intent")
    sentiment_by_intent = df.groupby(['intent', 'sentiment']).size().unstack(fill_value=0)
    
    fig = px.bar(
        sentiment_by_intent,
        title="Sentiment Distribution by Intent",
        labels={'value': 'Count', 'index': 'Intent'}
    )
    st.plotly_chart(fig, use_container_width=True)


# Add this to streamlit_app.py as a new function:

def show_performance_metrics():
    st.title("📈 Performance Metrics Analysis")
    
    # Load data
    @st.cache_data
    def load_performance_data():
        processor = DataProcessor()
        return processor.generate_sample_data(1000)
    
    df = load_performance_data()
    
    # Key Performance Indicators
    st.subheader("Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_resolution_time = df['Resolution_Time_Hours'].mean()
        st.metric("Avg Resolution Time", f"{avg_resolution_time:.1f}h")
    
    with col2:
        satisfaction_score = df['Customer_Satisfaction_Score'].mean()
        st.metric("Avg Satisfaction", f"{satisfaction_score:.1f}/5")
    
    with col3:
        timely_response_rate = (df['Timely_response'] == 'Yes').mean()
        st.metric("Timely Response Rate", f"{timely_response_rate:.1%}")
    
    with col4:
        resolution_rate = (df['Resolution_Status'] == 'Resolved').mean()
        st.metric("Resolution Rate", f"{resolution_rate:.1%}")
    
    with col5:
        escalation_rate = (df['Resolution_Status'] == 'Escalated').mean()
        st.metric("Escalation Rate", f"{escalation_rate:.1%}")
    
    # Detailed Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resolution Time Distribution")
        fig = px.histogram(
            df, x='Resolution_Time_Hours',
            bins=20,
            title="Distribution of Resolution Times"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Customer Satisfaction by Intent")
        satisfaction_by_intent = df.groupby('intent')['Customer_Satisfaction_Score'].mean()
        fig = px.bar(
            x=satisfaction_by_intent.index,
            y=satisfaction_by_intent.values,
            title="Average Satisfaction Score by Intent"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Resolution Status by Priority")
        resolution_priority = pd.crosstab(df['Priority_Level'], df['Resolution_Status'])
        fig = px.bar(
            resolution_priority,
            title="Resolution Status by Priority Level"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Response Time Analysis")
        avg_resolution_by_intent = df.groupby('intent')['Resolution_Time_Hours'].mean()
        fig = px.bar(
            x=avg_resolution_by_intent.index,
            y=avg_resolution_by_intent.values,
            title="Average Resolution Time by Intent"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic Analysis
    st.subheader("Geographic Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        state_performance = df.groupby('State').agg({
            'Resolution_Time_Hours': 'mean',
            'Customer_Satisfaction_Score': 'mean'
        }).round(2)
        st.dataframe(state_performance, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df, x='Resolution_Time_Hours', y='Customer_Satisfaction_Score',
            color='Priority_Level',
            title="Resolution Time vs Customer Satisfaction"
        )
        st.plotly_chart(fig, use_container_width=True)


# Add this as another page in your Streamlit app:

def show_geographic_analysis():
    st.title("🗺️ Geographic Analysis")
    
    @st.cache_data
    def load_geographic_data():
        processor = DataProcessor()
        return processor.generate_sample_data(2000)
    
    df = load_geographic_data()
    
    # State-wise analysis
    st.subheader("State-wise Performance Metrics")
    
    state_metrics = df.groupby('State').agg({
        'Complaint_ID': 'count',
        'Resolution_Time_Hours': 'mean',
        'Customer_Satisfaction_Score': 'mean',
        'Resolution_Status': lambda x: (x == 'Resolved').mean()
    }).round(2)
    
    state_metrics.columns = ['Total Complaints', 'Avg Resolution Time (hrs)', 
                           'Avg Satisfaction Score', 'Resolution Rate']
    
    st.dataframe(state_metrics, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Complaints by state
        state_counts = df['State'].value_counts()
        fig = px.bar(
            x=state_counts.values,
            y=state_counts.index,
            orientation='h',
            title="Total Complaints by State"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Average satisfaction by state
        fig = px.bar(
            x=state_metrics.index,
            y=state_metrics['Avg Satisfaction Score'],
            title="Average Customer Satisfaction by State"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Resolution time by state
        fig = px.bar(
            x=state_metrics.index,
            y=state_metrics['Avg Resolution Time (hrs)'],
            title="Average Resolution Time by State"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Intent distribution by state
        state_intent = pd.crosstab(df['State'], df['intent'])
        fig = px.bar(
            state_intent,
            title="Intent Distribution by State"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("Performance Heatmap")
    
    # Create correlation matrix for numeric columns by state
    numeric_cols = ['Resolution_Time_Hours', 'Customer_Satisfaction_Score']
    correlation_data = df.groupby('State')[numeric_cols].mean()
    
    fig = px.imshow(
        correlation_data.T,
        title="Performance Metrics Heatmap by State",
        color_continuous_scale='RdYlBu_r'
    )
    st.plotly_chart(fig, use_container_width=True)
