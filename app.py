# Installation requirements:
# pip install streamlit streamlit-chat streamlit-webrtc
# pip install pymupdf python-docx python-pptx pandas openpyxl xlrd striprtf beautifulsoup4
# pip install google-generativeai pinecone-client tiktoken scikit-learn numpy
# pip install pydub speechrecognition gtts

# main.py - Complete RAG Chatbot with Gemini & WebRTC
import streamlit as st
import sqlite3
import json
from datetime import datetime,date
import google.generativeai as genai


from typing import List, Dict, Any
import uuid

# WebRTC imports
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from Document_processor import DocumentProcessor, DocumentChunker
from vector_store import VectorStore
from LLM import LLMHandler
from Audio import VoiceHandler, AudioProcessor

class Setup:
    def __init__(self):
        self.setup_database()
        self.setup_apis()
        self.setup_accessibility()
    
    def setup_database(self):
        """Initialize SQLite for chat history"""
        # --- FIX: DeprecationWarning for datetime adapter in sqlite3 ---
        sqlite3.register_adapter(datetime, lambda ts: ts.isoformat())
        sqlite3.register_adapter(date, lambda d: d.isoformat()) # Also register date if you use date objects
        
        def convert_timestamp(val):
            return datetime.fromisoformat(val.decode('utf-8'))
        
        sqlite3.converters['DATETIME'] = convert_timestamp
        sqlite3.converters['TIMESTAMP'] = convert_timestamp # Add if you also use TIMESTAMP columns

        self.conn = sqlite3.connect('chat_history.db', check_same_thread=False,
                                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        # --- END FIX ---

        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                timestamp DATETIME,
                user_message TEXT,
                bot_response TEXT,
                context_docs TEXT,
                session_id TEXT
            )
        ''')
        self.conn.commit()
    

    
    def setup_apis(self):
        """Initialize API connections"""
        # Configure Gemini API
        if 'GEMINI_API_KEY' in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            st.success("✅ Gemini API configured")
        else:
            st.error("❌ Gemini API key not found in secrets")
        
        # Initialize components
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = VectorStore()
        if 'llm_handler' not in st.session_state:
            st.session_state.llm_handler = LLMHandler()
        if 'chunker' not in st.session_state:
            st.session_state.chunker = DocumentChunker()
    
    def setup_accessibility(self):
        """Configure accessibility settings"""
        if 'accessibility_settings' not in st.session_state:
            st.session_state.accessibility_settings = {
                'high_contrast': False,
                'large_text': False,
                'screen_reader_mode': False,
                'audio_speed': 1.0,
                'auto_play_responses': True
            }



class ContextManager:
    def __init__(self, db_connection):
        self.conn = db_connection
    
    def save_conversation(self, user_message: str, bot_response: str, 
                         context_docs: List[str], session_id: str):
        """Save conversation to database"""
        conversation_id = str(uuid.uuid4())
        self.conn.execute('''
            INSERT INTO conversations 
            (id, timestamp, user_message, bot_response, context_docs, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (conversation_id, datetime.now(), user_message, bot_response, 
              json.dumps(context_docs), session_id))
        self.conn.commit()
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve conversation history"""
        cursor = self.conn.execute('''
            SELECT user_message, bot_response, timestamp 
            FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (session_id, limit))
        
        return [{"user": row[0], "bot": row[1], "timestamp": row[2]} 
                for row in cursor.fetchall()]



def main():
    # Page config with accessibility
    st.set_page_config(
        page_title="Gemini RAG Chatbot with WebRTC",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = Setup()
        st.session_state.doc_processor = DocumentProcessor()
        st.session_state.voice_handler = VoiceHandler()
        st.session_state.context_manager = ContextManager(st.session_state.chatbot.conn)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.documents = {}
        st.session_state.webrtc_text = ""
    
    # Accessibility Controls Sidebar
    with st.sidebar:
        st.header("♿ Accessibility Settings")
        
        # Visual accessibility
        st.session_state.accessibility_settings['high_contrast'] = st.checkbox(
            "High Contrast Mode", 
            value=st.session_state.accessibility_settings['high_contrast']
        )
        
        st.session_state.accessibility_settings['large_text'] = st.checkbox(
            "Large Text Mode",
            value=st.session_state.accessibility_settings['large_text']
        )
        
        # Audio accessibility
        st.session_state.accessibility_settings['auto_play_responses'] = st.checkbox(
            "Auto-play Audio Responses",
            value=st.session_state.accessibility_settings['auto_play_responses']
        )
        
        st.divider()
        
        # Document Upload
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'pptx', 'txt'],
            help="Supported formats: PDF, Word, Excel, PowerPoint, Text"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.documents:
                    with st.spinner(f"Processing {file.name}..."):
                        text_content = st.session_state.doc_processor.process_document(file)
                        if text_content:
                            st.session_state.documents[file.name] = text_content
                            
                            with st.spinner(f"Creating Gemini embeddings for {file.name}..."):
                                chunks = st.session_state.chunker.chunk_text(text_content, file.name)
                                if chunks:
                                    success = st.session_state.vector_store.store_chunks(chunks)
                                    if success:
                                        st.success(f"✅ {file.name} processed and indexed! ({len(chunks)} chunks)")
                                    else:
                                        st.error(f"❌ Failed to index {file.name}")
                                else:
                                    st.warning(f"⚠️ No chunks created for {file.name}")
        
        # Show processed documents
        if st.session_state.documents:
            st.subheader("📚 Processed Documents")
            for doc_name in st.session_state.documents.keys():
                st.text(f"• {doc_name}")
    
    # Apply accessibility styles
    if st.session_state.accessibility_settings['high_contrast']:
        st.markdown("""
        <style>
        .stApp { background-color: black; color: white; }
        .stTextInput > div > div > input { background-color: #333; color: white; }
        </style>
        """, unsafe_allow_html=True)
    
    if st.session_state.accessibility_settings['large_text']:
        st.markdown("""
        <style>
        html, body, [class*="css"] { font-size: 18px !important; }
        </style>
        """, unsafe_allow_html=True)
    
    # Main Interface
    st.title("🎤 Gemini RAG Chatbot with WebRTC")
    st.caption("Voice-enabled document Q&A powered by Gemini 2.0 Flash")
    
    # Voice Input Section
    st.header("🎙️ Voice Input")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🌐 WebRTC Live Audio")
        
        # WebRTC Audio Streamer
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=AudioProcessor,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }),
            media_stream_constraints={"video": False, "audio": True},
            async_processing=True,
        )
        
        # Display recognized text from WebRTC
        if st.session_state.webrtc_text:
            st.success(f"🎯 Recognized: {st.session_state.webrtc_text}")
            
            if st.button("📝 Use WebRTC Text", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user", 
                    "content": st.session_state.webrtc_text
                })
                st.session_state.webrtc_text = ""  # Clear after use
                st.rerun()
    
    with col2:
        st.subheader("📝 Text Input")
        
        # Text input alternative
        text_input = st.text_input(
            "Type your question:", 
            key="text_question",
            help="You can type your question here as an alternative to voice input"
        )
        
        if text_input and st.button("🚀 Send Message", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": text_input})
            st.rerun()
    
    # Chat History Display
    st.header("💬 Conversation")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Add audio playback for bot responses
            if (message["role"] == "assistant" and 
                st.session_state.accessibility_settings['auto_play_responses']):
                audio_content = st.session_state.voice_handler.text_to_speech(message["content"])
                if audio_content:
                    st.audio(audio_content, format="audio/mp3")
    
    # Process new messages
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_question = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching relevant documents with Gemini embeddings..."):
                # Search for relevant chunks
                relevant_chunks = st.session_state.vector_store.similarity_search(
                    user_question, top_k=5
                )
                
                if relevant_chunks:
                    st.info(f"📚 Found {len(relevant_chunks)} relevant chunks from your documents")
                    sources = list(set([chunk['source'] for chunk in relevant_chunks]))
                    st.caption(f"Sources: {', '.join(sources)}")
                else:
                    st.warning("⚠️ No relevant content found in your documents")
                
                # Get conversation history
                history = st.session_state.context_manager.get_conversation_history(
                    st.session_state.session_id
                )
                
                with st.spinner("🤖 Generating response with Gemini 2.0 Flash..."):
                    # Generate response using Gemini
                    bot_response = st.session_state.llm_handler.generate_response(
                        user_question, 
                        relevant_chunks, 
                        history
                    )
                    
                    st.write(bot_response)
                    
                    # Save conversation
                    st.session_state.context_manager.save_conversation(
                        user_question, bot_response, 
                        [chunk['source'] for chunk in relevant_chunks],
                        st.session_state.session_id
                    )
                    
                    # Add response to messages
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    
                    # Generate and play audio response
                    if st.session_state.accessibility_settings['auto_play_responses']:
                        with st.spinner("🔊 Converting to speech..."):
                            audio_content = st.session_state.voice_handler.text_to_speech(bot_response)
                            if audio_content:
                                st.audio(audio_content, format="audio/mp3")
                
                # Show relevant chunks in expander
                if relevant_chunks:
                    with st.expander("📖 View Source Context"):
                        for i, chunk in enumerate(relevant_chunks):
                            st.markdown(f"""
                            **Source {i+1}: {chunk['source']}** (Similarity: {chunk['score']:.3f})
                            
                            {chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}
                            
                            ---
                            """)
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.webrtc_text = ""
        st.rerun()
    
    # Info Section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        **🚀 Features:**
        - **Gemini 2.5 Flash** for intelligent responses
        - **Gemini Text Embedding** for document search
        - **WebRTC** for real-time voice input
        - **Multi-format** document support (PDF, Word, Excel, PowerPoint)
        - **Accessibility** features with audio feedback
        - **Context-aware** conversations with memory
        
        **🎤 How to Use:**
        1. Upload your documents using the sidebar
        2. Use WebRTC for real-time voice input or type your questions
        3. Get intelligent responses based on your documents
        4. Enjoy audio feedback for accessibility
        
        **🔧 Powered by:**
        - Google Gemini 2.0 Flash (LLM)
        - Google Text Embedding (Vector Search)
        - Streamlit WebRTC (Voice Input)
        - Pinecone (Vector Database)
        """)

if __name__ == "__main__":
    main()