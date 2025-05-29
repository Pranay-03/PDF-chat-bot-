import google.generativeai as genai
from typing import List, Dict, Any
import streamlit as st

class LLMHandler:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]], 
                         conversation_history: List[Dict[str, str]]) -> str:
        """Generate response using Gemini 2.5 Flash"""
        
        # Build context from retrieved chunks
        context_text = "\n\n".join([
            f"Source: {chunk['source']}\nContent: {chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            conversation_context = "Previous conversation:\n"
            for msg in conversation_history[-3:]:
                conversation_context += f"User: {msg['user']}\nAssistant: {msg['bot']}\n\n"
        
        # Create comprehensive prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on provided documents and conversation context.

Guidelines:
1. Use the provided document context to answer questions accurately
2. Reference specific sources when making claims
3. If the information isn't in the documents, say so clearly
4. Consider the conversation history for context
5. Be conversational and helpful
6. If asked about previous topics, refer to the conversation history

{conversation_context}

Document Context:
{context_text}

Current Question: {query}

Please provide a comprehensive answer based on the document context and conversation history."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Gemini API error: {str(e)}")
            return "I apologize, but I'm having trouble generating a response. Please try again."