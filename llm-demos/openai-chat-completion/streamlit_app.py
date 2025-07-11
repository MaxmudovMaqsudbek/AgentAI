"""
Advanced Business AI Assistant - Streamlit App
Deployed on Hugging Face Spaces

This comprehensive business assistant provides intelligent answers about your business,
integrates with multiple AI providers, and includes support ticket functionality.
"""

import streamlit as st
import time
import json
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid
import hashlib

# Import the streaming system
try:
    from streaming import AdvancedStreamingSystem, StreamProvider, StreamingConfig
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AgentAI Business Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Business Information Database
BUSINESS_INFO = {
    "company_name": "AgentAI Solutions",
    "tagline": "Advanced AI Development & Consulting",
    "description": "We specialize in cutting-edge AI solutions, machine learning implementations, and intelligent automation systems for businesses worldwide.",
    
    "contact": {
        "phone": "+1 (555) 123-4567",
        "email": "info@agentai-solutions.com",
        "support_email": "support@agentai-solutions.com",
        "website": "https://agentai-solutions.com",
        "linkedin": "https://linkedin.com/company/agentai-solutions"
    },
    
    "location": {
        "address": "123 Innovation Drive, Tech District",
        "city": "San Francisco",
        "state": "CA",
        "zip": "94105",
        "country": "USA",
        "timezone": "PST (UTC-8)",
        "coordinates": {"lat": 37.7749, "lon": -122.4194}
    },
    
    "hours": {
        "monday": {"open": "9:00 AM", "close": "6:00 PM", "timezone": "PST"},
        "tuesday": {"open": "9:00 AM", "close": "6:00 PM", "timezone": "PST"},
        "wednesday": {"open": "9:00 AM", "close": "6:00 PM", "timezone": "PST"},
        "thursday": {"open": "9:00 AM", "close": "6:00 PM", "timezone": "PST"},
        "friday": {"open": "9:00 AM", "close": "6:00 PM", "timezone": "PST"},
        "saturday": {"open": "10:00 AM", "close": "4:00 PM", "timezone": "PST"},
        "sunday": "Closed",
        "holidays": "Closed on major US holidays"
    },
    
    "services": [
        {
            "name": "AI Development",
            "description": "Custom AI solutions, machine learning models, and intelligent automation systems",
            "duration": "2-12 weeks",
            "price_range": "$10,000 - $100,000+"
        },
        {
            "name": "AI Consulting",
            "description": "Strategic AI planning, feasibility studies, and implementation roadmaps",
            "duration": "1-4 weeks",
            "price_range": "$5,000 - $25,000"
        },
        {
            "name": "Model Training",
            "description": "Custom model training, fine-tuning, and optimization services",
            "duration": "1-6 weeks",
            "price_range": "$3,000 - $50,000"
        },
        {
            "name": "AI Integration",
            "description": "Seamless integration of AI solutions into existing business systems",
            "duration": "2-8 weeks",
            "price_range": "$7,500 - $75,000"
        }
    ],
    
    "team": [
        {"name": "Alex Chen", "role": "CEO & AI Architect", "expertise": "Machine Learning, Deep Learning"},
        {"name": "Sarah Johnson", "role": "CTO & Lead Developer", "expertise": "MLOps, Cloud Architecture"},
        {"name": "Dr. Michael Rodriguez", "role": "Head of Research", "expertise": "NLP, Computer Vision"},
        {"name": "Emily Zhang", "role": "AI Consultant", "expertise": "Business Strategy, AI Ethics"}
    ],
    
    "support": {
        "response_time": "24 hours for general inquiries, 4 hours for urgent issues",
        "channels": ["Email", "Phone", "Live Chat", "Support Tickets"],
        "emergency_contact": "+1 (555) 123-4567 ext. 911"
    }
}

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .business-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-left: 4px solid #2196F3;
        margin-left: 2rem;
    }
    .assistant-message {
        background: linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%);
        border-left: 4px solid #9C27B0;
        margin-right: 2rem;
    }
    .support-ticket {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFCC80 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-open { background-color: #4CAF50; }
    .status-closed { background-color: #F44336; }
    .status-pending { background-color: #FF9800; }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.8rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
def initialize_session_state():
    """Initialize session state variables."""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    if 'support_tickets' not in st.session_state:
        st.session_state.support_tickets = []
    
    if 'business_assistant' not in st.session_state:
        st.session_state.business_assistant = None
    
    if 'user_session_id' not in st.session_state:
        st.session_state.user_session_id = str(uuid.uuid4())
    
    if 'chat_analytics' not in st.session_state:
        st.session_state.chat_analytics = []

class BusinessKnowledgeBase:
    """Advanced business knowledge base with intelligent query processing."""
    
    def __init__(self, business_info: Dict):
        self.business_info = business_info
        self.knowledge_patterns = self._build_knowledge_patterns()
    
    def _build_knowledge_patterns(self) -> Dict[str, Any]:
        """Build patterns for intelligent query matching."""
        return {
            "hours": {
                "keywords": ["hours", "open", "close", "time", "schedule", "when", "operating", "business hours", "opening", "closing"],
                "context": "business hours and schedule"
            },
            "location": {
                "keywords": ["location", "address", "where", "directions", "map", "office", "find", "located", "place"],
                "context": "business location and address"
            },
            "contact": {
                "keywords": ["phone", "email", "contact", "call", "reach", "number", "telephone", "reach", "communicate"],
                "context": "contact information"
            },
            "services": {
                "keywords": ["services", "what", "offer", "do", "provide", "solutions", "products", "work", "help", "ai", "development"],
                "context": "services and offerings"
            },
            "team": {
                "keywords": ["team", "staff", "who", "people", "employees", "founders", "ceo", "cto", "about"],
                "context": "team information"
            },
            "support": {
                "keywords": ["support", "help", "assistance", "problem", "issue", "ticket", "emergency"],
                "context": "support and assistance"
            },
            "pricing": {
                "keywords": ["price", "cost", "pricing", "fee", "rates", "budget", "expensive", "cheap", "money"],
                "context": "pricing information"
            },
            "general": {
                "keywords": ["hello", "hi", "hey", "what", "who", "company", "business", "about"],
                "context": "general information"
            }
        }
    
    def search_knowledge(self, query: str) -> Dict[str, Any]:
        """Search knowledge base for relevant information."""
        query_lower = query.lower()
        matches = {}
        
        for category, pattern in self.knowledge_patterns.items():
            score = sum(1 for keyword in pattern["keywords"] if keyword in query_lower)
            if score > 0:
                matches[category] = {
                    "score": score,
                    "context": pattern["context"],
                    "data": self._get_category_data(category)
                }
        
        return dict(sorted(matches.items(), key=lambda x: x[1]["score"], reverse=True))
    
    def _get_category_data(self, category: str) -> Any:
        """Get data for a specific category."""
        category_mapping = {
            "hours": self.business_info["hours"],
            "location": self.business_info["location"],
            "contact": self.business_info["contact"],
            "services": self.business_info["services"],
            "team": self.business_info["team"],
            "support": self.business_info["support"],
            "pricing": [service for service in self.business_info["services"]]
        }
        return category_mapping.get(category, {})
    
    def generate_business_context(self) -> str:
        """Generate comprehensive business context for AI queries."""
        context_parts = [
            f"Company: {self.business_info['company_name']}",
            f"Description: {self.business_info['description']}",
            f"Location: {self.business_info['location']['address']}, {self.business_info['location']['city']}, {self.business_info['location']['state']}",
            f"Phone: {self.business_info['contact']['phone']}",
            f"Email: {self.business_info['contact']['email']}",
            f"Website: {self.business_info['contact']['website']}",
            "",
            "Business Hours:",
        ]
        
        for day, hours in self.business_info["hours"].items():
            if day != "holidays":
                if isinstance(hours, dict):
                    context_parts.append(f"  {day.title()}: {hours['open']} - {hours['close']} {hours['timezone']}")
                else:
                    context_parts.append(f"  {day.title()}: {hours}")
        
        context_parts.extend([
            "",
            "Services:",
        ])
        
        for service in self.business_info["services"]:
            context_parts.append(f"  - {service['name']}: {service['description']} (Duration: {service['duration']}, Price: {service['price_range']})")
        
        return "\n".join(context_parts)

class SupportTicketManager:
    """Advanced support ticket management system."""
    
    def __init__(self):
        self.ticket_categories = [
            "General Inquiry",
            "Technical Support",
            "Billing Question",
            "Service Request",
            "Bug Report",
            "Feature Request",
            "Emergency"
        ]
        self.priorities = ["Low", "Medium", "High", "Critical"]
    
    def create_ticket(self, title: str, description: str, category: str, priority: str, 
                     contact_info: Dict, user_session_id: str) -> Dict[str, Any]:
        """Create a new support ticket."""
        ticket = {
            "id": self._generate_ticket_id(),
            "title": title,
            "description": description,
            "category": category,
            "priority": priority,
            "status": "Open",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "contact_info": contact_info,
            "user_session_id": user_session_id,
            "responses": []
        }
        
        # Simulate external ticket system integration
        self._integrate_with_external_systems(ticket)
        
        return ticket
    
    def _generate_ticket_id(self) -> str:
        """Generate unique ticket ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"TKT-{timestamp}-{random_suffix.upper()}"
    
    def _integrate_with_external_systems(self, ticket: Dict) -> bool:
        """Simulate integration with external ticket systems."""
        try:
            # Simulate GitHub Issues integration
            github_issue = self._create_github_issue(ticket)
            
            # Simulate Jira integration
            jira_ticket = self._create_jira_ticket(ticket)
            
            # Simulate Trello integration
            trello_card = self._create_trello_card(ticket)
            
            ticket["external_integrations"] = {
                "github": github_issue,
                "jira": jira_ticket,
                "trello": trello_card
            }
            
            return True
            
        except Exception as e:
            st.warning(f"External integration warning: {e}")
            return False
    
    def _create_github_issue(self, ticket: Dict) -> Dict:
        """Simulate GitHub issue creation."""
        return {
            "platform": "GitHub",
            "url": f"https://github.com/agentai-solutions/support/issues/{ticket['id']}",
            "issue_number": len(st.session_state.support_tickets) + 1,
            "status": "created",
            "labels": [ticket["category"].lower().replace(" ", "-"), ticket["priority"].lower()]
        }
    
    def _create_jira_ticket(self, ticket: Dict) -> Dict:
        """Simulate Jira ticket creation."""
        return {
            "platform": "Jira",
            "url": f"https://agentai-solutions.atlassian.net/browse/{ticket['id']}",
            "key": ticket['id'],
            "status": "Open",
            "project": "SUPPORT"
        }
    
    def _create_trello_card(self, ticket: Dict) -> Dict:
        """Simulate Trello card creation."""
        return {
            "platform": "Trello",
            "url": f"https://trello.com/c/{ticket['id'][:8]}",
            "board": "Customer Support",
            "list": "New Tickets",
            "card_id": ticket['id'][:8]
        }

class BusinessAIAssistant:
    """Advanced AI-powered business assistant."""
    
    def __init__(self):
        self.knowledge_base = BusinessKnowledgeBase(BUSINESS_INFO)
        self.ticket_manager = SupportTicketManager()
        
        if STREAMING_AVAILABLE:
            try:
                self.streaming_system = AdvancedStreamingSystem()
                self.ai_available = True
            except Exception as e:
                st.warning(f"AI streaming not available: {e}")
                self.ai_available = False
        else:
            self.ai_available = False
    
    def process_query(self, query: str, use_ai: bool = True) -> Dict[str, Any]:
        """Process user query with knowledge base and AI assistance."""
        # Search knowledge base
        knowledge_matches = self.knowledge_base.search_knowledge(query)
        
        # Always try to provide a helpful response
        if use_ai and self.ai_available:
            try:
                ai_response = self._get_ai_response(query, knowledge_matches)
                return {
                    "type": "ai_enhanced",
                    "response": ai_response,
                    "knowledge_matches": knowledge_matches,
                    "confidence": "high"
                }
            except Exception as e:
                # Fall back to enhanced response
                pass
        
        # Use enhanced response logic
        enhanced_response = self._create_enhanced_response(query, knowledge_matches)
        return {
            "type": "enhanced",
            "response": enhanced_response,
            "knowledge_matches": knowledge_matches,
            "confidence": "medium" if knowledge_matches else "low"
        }
    
    def _get_ai_response(self, query: str, knowledge_matches: Dict) -> str:
        """Get AI-enhanced response using streaming system."""
        try:
            # Create context-aware prompt
            business_context = self.knowledge_base.generate_business_context()
            
            # Prepare AI messages with more specific instructions
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a helpful business assistant for {BUSINESS_INFO['company_name']}. 
                    
Business Information:
{business_context}

Instructions:
- Always provide helpful responses based on the business information above
- Be professional, friendly, and informative
- Use the business hours, contact info, location, and services from the context
- If asked about something not in the business info, acknowledge it politely and suggest contacting support
- Keep responses conversational but informative
- Always try to be helpful even for general questions"""
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
            
            # Use streaming system for AI response
            if self.streaming_system and hasattr(self.streaming_system, 'streamers') and self.streaming_system.streamers:
                provider = list(self.streaming_system.streamers.keys())[0]
                config = StreamingConfig(
                    model="gemini-1.5-flash",
                    temperature=0.3,
                    max_tokens=500,
                    show_timing=False
                )
                
                # Use a simpler streaming approach for Streamlit
                streamer = self.streaming_system.streamers[provider]
                content_parts = []
                
                for chunk in streamer.stream_completion(messages, config):
                    content_parts.append(chunk.content)
                
                full_response = ''.join(content_parts)
                if full_response.strip():
                    return full_response.strip()
            
        except Exception as e:
            # Log the error but don't show to user
            import logging
            logging.warning(f"AI response error: {e}")
        
        # Enhanced fallback response
        return self._create_enhanced_response(query, knowledge_matches)
    
    def _create_enhanced_response(self, query: str, knowledge_matches: Dict) -> str:
        """Create enhanced response with better query understanding."""
        query_lower = query.lower()
        
        # Always try to provide helpful information
        if knowledge_matches:
            return self._create_structured_response(knowledge_matches)
        
        # Smart fallback responses based on query patterns
        if any(word in query_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return f"Hello! Welcome to {BUSINESS_INFO['company_name']}. How can I help you today? You can ask about our business hours, location, services, or contact information."
        
        elif any(word in query_lower for word in ['help', 'assist', 'support']):
            return f"I'm here to help! I can provide information about {BUSINESS_INFO['company_name']} including:\n\n• 🕐 Business hours and schedule\n• 📍 Location and directions\n• 💼 Services and offerings\n• 📞 Contact information\n• 👥 Our team\n\nWhat would you like to know?"
        
        elif any(word in query_lower for word in ['who', 'what', 'company', 'business']):
            return f"**About {BUSINESS_INFO['company_name']}**\n\n{BUSINESS_INFO['description']}\n\n📍 **Location:** {BUSINESS_INFO['location']['city']}, {BUSINESS_INFO['location']['state']}\n📞 **Phone:** {BUSINESS_INFO['contact']['phone']}\n📧 **Email:** {BUSINESS_INFO['contact']['email']}\n\nWould you like to know more about our services or how to contact us?"
        
        elif any(word in query_lower for word in ['thank', 'thanks']):
            return f"You're welcome! Is there anything else you'd like to know about {BUSINESS_INFO['company_name']}? I'm here to help with information about our services, hours, location, or anything else!"
        
        else:
            # General helpful response
            return f"I'd be happy to help you with information about {BUSINESS_INFO['company_name']}! While I don't have specific information about '{query}', I can help you with:\n\n• 🕐 **Business Hours:** When we're open\n• 📍 **Location:** Where to find us\n• 💼 **Services:** What we offer\n• 📞 **Contact:** How to reach us\n• 👥 **Team:** Who we are\n\nWhat would you like to know more about?"
    
    def _create_structured_response(self, knowledge_matches: Dict) -> str:
        """Create structured response from knowledge base matches."""
        if not knowledge_matches:
            return "I don't have specific information about that. Please contact our support team for assistance."
        
        response_parts = []
        
        for category, match in list(knowledge_matches.items())[:2]:  # Top 2 matches
            data = match["data"]
            
            if category == "hours":
                response_parts.append("📅 **Business Hours:**")
                for day, hours in data.items():
                    if day != "holidays":
                        if isinstance(hours, dict):
                            response_parts.append(f"  • {day.title()}: {hours['open']} - {hours['close']} {hours['timezone']}")
                        else:
                            response_parts.append(f"  • {day.title()}: {hours}")
            
            elif category == "location":
                response_parts.append(f"📍 **Location:**")
                response_parts.append(f"  • Address: {data['address']}")
                response_parts.append(f"  • City: {data['city']}, {data['state']} {data['zip']}")
                response_parts.append(f"  • Timezone: {data['timezone']}")
            
            elif category == "contact":
                response_parts.append("📞 **Contact Information:**")
                response_parts.append(f"  • Phone: {data['phone']}")
                response_parts.append(f"  • Email: {data['email']}")
                response_parts.append(f"  • Website: {data['website']}")
            
            elif category == "services":
                response_parts.append("💼 **Our Services:**")
                for service in data[:3]:  # Top 3 services
                    response_parts.append(f"  • **{service['name']}**: {service['description']}")
            
            response_parts.append("")  # Add spacing
        
        return "\n".join(response_parts)

def display_business_header():
    """Display business header with key information."""
    st.markdown(f'<h1 class="main-header">🏢 {BUSINESS_INFO["company_name"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{BUSINESS_INFO["tagline"]}</p>', unsafe_allow_html=True)
    
    # Business card
    st.markdown(f"""
    <div class="business-card">
        <h3>📍 Quick Info</h3>
        <p><strong>📞 Phone:</strong> {BUSINESS_INFO["contact"]["phone"]}</p>
        <p><strong>📧 Email:</strong> {BUSINESS_INFO["contact"]["email"]}</p>
        <p><strong>🌐 Website:</strong> {BUSINESS_INFO["contact"]["website"]}</p>
        <p><strong>📍 Location:</strong> {BUSINESS_INFO["location"]["city"]}, {BUSINESS_INFO["location"]["state"]}</p>
    </div>
    """, unsafe_allow_html=True)

def display_business_status():
    """Display current business status and metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🕐 Status</h4>
            <span class="status-indicator status-open"></span>
            <strong>Open Now</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>⏱️ Response Time</h4>
            <strong>< 24 hours</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        ticket_count = len(st.session_state.support_tickets)
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎫 Active Tickets</h4>
            <strong>{ticket_count}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 AI Status</h4>
            <strong>Online</strong>
        </div>
        """, unsafe_allow_html=True)

def create_sidebar():
    """Create sidebar with navigation and options."""
    st.sidebar.title("🏢 Business Assistant")
    
    # Navigation
    page = st.sidebar.selectbox(
        "📋 Navigate",
        ["💬 Chat Assistant", "🎫 Support Tickets", "📊 Business Info", "⚙️ Settings"]
    )
    
    # Quick actions
    st.sidebar.markdown("### 🚀 Quick Actions")
    
    if st.sidebar.button("📞 Call Now", use_container_width=True):
        st.sidebar.success(f"Call: {BUSINESS_INFO['contact']['phone']}")
    
    if st.sidebar.button("📧 Send Email", use_container_width=True):
        st.sidebar.success(f"Email: {BUSINESS_INFO['contact']['email']}")
    
    if st.sidebar.button("🌐 Visit Website", use_container_width=True):
        st.sidebar.success(f"Website: {BUSINESS_INFO['contact']['website']}")
    
    # Business hours
    st.sidebar.markdown("### 🕐 Today's Hours")
    current_day = datetime.now().strftime("%A").lower()
    today_hours = BUSINESS_INFO["hours"].get(current_day, "Closed")
    
    if isinstance(today_hours, dict):
        st.sidebar.info(f"**{current_day.title()}:** {today_hours['open']} - {today_hours['close']}")
    else:
        st.sidebar.info(f"**{current_day.title()}:** {today_hours}")
    
    return page

def chat_assistant_page():
    """Main chat assistant interface."""
    st.markdown("### 💬 Business Assistant Chat")
    st.markdown("Ask me anything about our business - hours, location, services, or contact information!")
    
    # Initialize assistant
    if st.session_state.business_assistant is None:
        st.session_state.business_assistant = BusinessAIAssistant()
    
    assistant = st.session_state.business_assistant
    
    # Display conversation
    for message in st.session_state.conversation:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your question:",
            placeholder="Ask about our hours, location, services, contact info...",
            height=100
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            send_button = st.form_submit_button("🚀 Send", use_container_width=True)
        
        with col2:
            ai_enabled = st.form_submit_button("🤖 AI Mode", use_container_width=True)
        
        with col3:
            clear_button = st.form_submit_button("🗑️ Clear", use_container_width=True)
    
    # Process input
    if (send_button or ai_enabled) and user_input.strip():
        # Add user message
        st.session_state.conversation.append({
            "role": "user",
            "content": user_input.strip()
        })
        
        # Process query
        with st.spinner("🤖 Processing your question..."):
            response_data = assistant.process_query(user_input.strip(), use_ai=ai_enabled)
        
        # Add assistant response
        st.session_state.conversation.append({
            "role": "assistant",
            "content": response_data["response"]
        })
        
        # Show confidence and suggest ticket if needed
        if response_data["confidence"] == "low":
            st.warning("⚠️ I couldn't find specific information about that. Would you like to create a support ticket?")
            if st.button("🎫 Create Support Ticket"):
                st.session_state.pending_ticket_query = user_input.strip()
                st.rerun()
        
        st.rerun()
    
    if clear_button:
        st.session_state.conversation = []
        st.rerun()

def support_tickets_page():
    """Support ticket management interface."""
    st.markdown("### 🎫 Support Ticket System")
    
    tab1, tab2 = st.tabs(["📝 Create Ticket", "📋 My Tickets"])
    
    with tab1:
        st.markdown("#### Create New Support Ticket")
        
        if st.session_state.business_assistant is None:
            st.session_state.business_assistant = BusinessAIAssistant()
        
        ticket_manager = st.session_state.business_assistant.ticket_manager
        
        with st.form("create_ticket_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Ticket Title*", placeholder="Brief description of your issue")
                category = st.selectbox("Category*", ticket_manager.ticket_categories)
                priority = st.selectbox("Priority*", ticket_manager.priorities)
            
            with col2:
                name = st.text_input("Your Name*", placeholder="Full name")
                email = st.text_input("Email*", placeholder="your.email@example.com")
                phone = st.text_input("Phone", placeholder="Optional phone number")
            
            description = st.text_area(
                "Description*",
                placeholder="Detailed description of your issue or request...",
                height=150
            )
            
            # Pre-fill if coming from chat
            if hasattr(st.session_state, 'pending_ticket_query'):
                description = st.text_area(
                    "Description*",
                    value=st.session_state.pending_ticket_query,
                    height=150
                )
                del st.session_state.pending_ticket_query
            
            submit_ticket = st.form_submit_button("🎫 Create Ticket", use_container_width=True)
            
            if submit_ticket:
                if title and description and name and email and category and priority:
                    # Create ticket
                    contact_info = {"name": name, "email": email, "phone": phone}
                    
                    ticket = ticket_manager.create_ticket(
                        title, description, category, priority,
                        contact_info, st.session_state.user_session_id
                    )
                    
                    # Add to session
                    st.session_state.support_tickets.append(ticket)
                    
                    # Display success
                    st.success(f"✅ Ticket created successfully!")
                    st.markdown(f"""
                    <div class="support-ticket">
                        <h4>🎫 Ticket #{ticket['id']}</h4>
                        <p><strong>Status:</strong> <span class="status-indicator status-open"></span> {ticket['status']}</p>
                        <p><strong>Priority:</strong> {ticket['priority']}</p>
                        <p><strong>Category:</strong> {ticket['category']}</p>
                        <p><strong>Created:</strong> {ticket['created_at'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                        
                        <h5>External Integrations:</h5>
                        <ul>
                            <li>📚 GitHub: <a href="{ticket['external_integrations']['github']['url']}" target="_blank">Issue #{ticket['external_integrations']['github']['issue_number']}</a></li>
                            <li>🔧 Jira: <a href="{ticket['external_integrations']['jira']['url']}" target="_blank">{ticket['external_integrations']['jira']['key']}</a></li>
                            <li>📋 Trello: <a href="{ticket['external_integrations']['trello']['url']}" target="_blank">Card {ticket['external_integrations']['trello']['card_id']}</a></li>
                        </ul>
                        
                        <p><strong>Expected Response:</strong> {BUSINESS_INFO['support']['response_time']}</p>
                        <p>You will receive updates at: <strong>{email}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.error("❌ Please fill in all required fields marked with *")
    
    with tab2:
        st.markdown("#### Your Support Tickets")
        
        if st.session_state.support_tickets:
            for ticket in reversed(st.session_state.support_tickets):  # Most recent first
                status_class = f"status-{ticket['status'].lower()}"
                
                st.markdown(f"""
                <div class="support-ticket">
                    <h4>🎫 {ticket['title']}</h4>
                    <p><strong>ID:</strong> {ticket['id']}</p>
                    <p><strong>Status:</strong> <span class="status-indicator {status_class}"></span> {ticket['status']}</p>
                    <p><strong>Priority:</strong> {ticket['priority']} | <strong>Category:</strong> {ticket['category']}</p>
                    <p><strong>Created:</strong> {ticket['created_at'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Description:</strong> {ticket['description'][:200]}...</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📝 No support tickets yet. Create one above if you need assistance!")

def business_info_page():
    """Business information dashboard."""
    st.markdown("### 📊 Business Information")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏢 Overview", "🕐 Hours", "💼 Services", "👥 Team"])
    
    with tab1:
        st.markdown("#### Company Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="info-card">
                <h4>🏢 {BUSINESS_INFO['company_name']}</h4>
                <p><strong>Description:</strong> {BUSINESS_INFO['description']}</p>
                <p><strong>Tagline:</strong> {BUSINESS_INFO['tagline']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-card">
                <h4>📍 Location</h4>
                <p><strong>Address:</strong> {BUSINESS_INFO['location']['address']}</p>
                <p><strong>City:</strong> {BUSINESS_INFO['location']['city']}, {BUSINESS_INFO['location']['state']} {BUSINESS_INFO['location']['zip']}</p>
                <p><strong>Country:</strong> {BUSINESS_INFO['location']['country']}</p>
                <p><strong>Timezone:</strong> {BUSINESS_INFO['location']['timezone']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card">
                <h4>📞 Contact Information</h4>
                <p><strong>Phone:</strong> {BUSINESS_INFO['contact']['phone']}</p>
                <p><strong>Email:</strong> {BUSINESS_INFO['contact']['email']}</p>
                <p><strong>Support Email:</strong> {BUSINESS_INFO['contact']['support_email']}</p>
                <p><strong>Website:</strong> <a href="{BUSINESS_INFO['contact']['website']}" target="_blank">{BUSINESS_INFO['contact']['website']}</a></p>
                <p><strong>LinkedIn:</strong> <a href="{BUSINESS_INFO['contact']['linkedin']}" target="_blank">Company Profile</a></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-card">
                <h4>🎯 Support Information</h4>
                <p><strong>Response Time:</strong> {BUSINESS_INFO['support']['response_time']}</p>
                <p><strong>Channels:</strong> {', '.join(BUSINESS_INFO['support']['channels'])}</p>
                <p><strong>Emergency:</strong> {BUSINESS_INFO['support']['emergency_contact']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### Business Hours")
        
        hours_data = []
        for day, hours in BUSINESS_INFO["hours"].items():
            if day != "holidays":
                if isinstance(hours, dict):
                    hours_data.append({
                        "Day": day.title(),
                        "Open": hours["open"],
                        "Close": hours["close"],
                        "Timezone": hours["timezone"]
                    })
                else:
                    hours_data.append({
                        "Day": day.title(),
                        "Open": hours,
                        "Close": "",
                        "Timezone": ""
                    })
        
        st.table(hours_data)
        
        st.info(f"🎄 Holiday Hours: {BUSINESS_INFO['hours']['holidays']}")
    
    with tab3:
        st.markdown("#### Our Services")
        
        for service in BUSINESS_INFO["services"]:
            st.markdown(f"""
            <div class="info-card">
                <h4>💼 {service['name']}</h4>
                <p><strong>Description:</strong> {service['description']}</p>
                <p><strong>Duration:</strong> {service['duration']}</p>
                <p><strong>Price Range:</strong> {service['price_range']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("#### Our Team")
        
        cols = st.columns(2)
        for i, member in enumerate(BUSINESS_INFO["team"]):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="info-card">
                    <h4>👤 {member['name']}</h4>
                    <p><strong>Role:</strong> {member['role']}</p>
                    <p><strong>Expertise:</strong> {member['expertise']}</p>
                </div>
                """, unsafe_allow_html=True)

def settings_page():
    """Settings and configuration page."""
    st.markdown("### ⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🤖 AI Configuration")
        
        ai_enabled = st.checkbox("Enable AI Responses", value=True)
        ai_provider = st.selectbox("AI Provider", ["Auto", "OpenAI", "Gemini", "Hugging Face"])
        response_length = st.slider("Response Length", 100, 1000, 500)
        
        st.markdown("#### 🎫 Ticket Settings")
        auto_ticket = st.checkbox("Auto-suggest tickets for unknown queries", value=True)
        email_notifications = st.checkbox("Email notifications", value=True)
        
    with col2:
        st.markdown("#### 📊 Analytics")
        
        st.metric("Total Conversations", len(st.session_state.conversation))
        st.metric("Support Tickets Created", len(st.session_state.support_tickets))
        st.metric("Session ID", st.session_state.user_session_id[:8] + "...")
        
        st.markdown("#### 🔄 Actions")
        
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.support_tickets = []
            st.session_state.chat_analytics = []
            st.success("✅ All data cleared!")
        
        if st.button("💾 Export Data", use_container_width=True):
            export_data = {
                "conversation": st.session_state.conversation,
                "support_tickets": [
                    {**ticket, "created_at": ticket["created_at"].isoformat(), 
                     "updated_at": ticket["updated_at"].isoformat()}
                    for ticket in st.session_state.support_tickets
                ],
                "analytics": st.session_state.chat_analytics,
                "session_id": st.session_state.user_session_id,
                "export_timestamp": datetime.now().isoformat()
            }
            
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"business_assistant_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_business_header()
    
    # Display status
    display_business_status()
    
    # Create sidebar and get current page
    current_page = create_sidebar()
    
    # Display appropriate page
    st.markdown("---")
    
    if current_page == "💬 Chat Assistant":
        chat_assistant_page()
    elif current_page == "🎫 Support Tickets":
        support_tickets_page()
    elif current_page == "📊 Business Info":
        business_info_page()
    elif current_page == "⚙️ Settings":
        settings_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"🚀 **{BUSINESS_INFO['company_name']}** | "
        f"📞 {BUSINESS_INFO['contact']['phone']} | "
        f"📧 {BUSINESS_INFO['contact']['email']} | "
        f"Powered by Advanced AI & Hugging Face Spaces"
    )

if __name__ == "__main__":
    main()
