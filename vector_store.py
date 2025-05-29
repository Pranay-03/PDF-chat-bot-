import streamlit as st
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from typing import List, Dict, Any

import time

import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    def __init__(self):
        self.pc = None
        self.index = None
        self.local_vectors = {}
        self.setup_pinecone()
    
    def setup_pinecone(self):
        """Initialize Pinecone vector database with new API"""
        try:
            if 'PINECONE_API_KEY' in st.secrets:
                # Initialize Pinecone client
                self.pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
                
                index_name = "rag-chatbot-gemini"
                
                # Check if index exists
                existing_indexes = [index.name for index in self.pc.list_indexes()]
                
                if index_name not in existing_indexes:
                    # Create index with new API
                    self.pc.create_index(
                        name=index_name,
                        dimension=768,  # Gemini embedding dimension
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        )
                    )
                    # Wait for index to be ready
                    time.sleep(10)
                
                # Connect to index
                self.index = self.pc.Index(index_name)
                st.success("✅ Connected to Pinecone vector database")
                
        except Exception as e:
            st.warning(f"Pinecone setup failed, using local storage: {str(e)}")
            self.pc = None
            self.index = None
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Gemini API"""
        try:
            embeddings = []
            for text in texts:
                # Use Gemini's embedding model
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            st.error(f"Gemini embedding generation failed: {str(e)}")
            return []
    
    def store_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Store document chunks with embeddings"""
        if not chunks:
            return False
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        if not embeddings:
            return False
        
        # Store in Pinecone if available
        if self.index:
            try:
                vectors_to_upsert = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    vectors_to_upsert.append({
                        'id': chunk['chunk_id'],
                        'values': embedding,
                        'metadata': {
                            'text': chunk['text'],
                            'source': chunk['source'],
                            'tokens': chunk['tokens']
                        }
                    })
                
                # Upsert in batches
                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i + batch_size]
                    self.index.upsert(vectors=batch)
                
                return True
            except Exception as e:
                st.error(f"Pinecone storage failed: {str(e)}")
        
        # Fallback to local storage
        for chunk, embedding in zip(chunks, embeddings):
            self.local_vectors[chunk['chunk_id']] = {
                'embedding': embedding,
                'metadata': {
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'tokens': chunk['tokens']
                }
            }
        
        return True
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        try:
            query_embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )['embedding']
        except Exception as e:
            st.error(f"Query embedding failed: {str(e)}")
            return []
        
        # Search in Pinecone if available
        if self.index:
            try:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                
                return [{
                    'text': match['metadata']['text'],
                    'source': match['metadata']['source'],
                    'score': match['score'],
                    'tokens': match['metadata'].get('tokens', 0)
                } for match in results['matches']]
            except Exception as e:
                st.error(f"Pinecone search failed: {str(e)}")
        
        # Fallback to local search
        if not self.local_vectors:
            return []
        
        similarities = []
        for chunk_id, data in self.local_vectors.items():
            similarity = cosine_similarity(
                [query_embedding], 
                [data['embedding']]
            )[0][0]
            
            similarities.append({
                'text': data['metadata']['text'],
                'source': data['metadata']['source'],
                'score': similarity,
                'tokens': data['metadata']['tokens'],
                'chunk_id': chunk_id
            })
        
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]