import numpy as np
import speech_recognition as sr
from gtts import gTTS
import io
import tempfile
import os

import streamlit as st
# WebRTC imports
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import pydub
from pydub import AudioSegment


class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def process_audio_frame(self, audio_frame):
        """Process audio frame from WebRTC"""
        try:
            # Convert audio frame to wav
            audio_segment = AudioSegment(
                audio_frame.to_ndarray().tobytes(),
                frame_rate=audio_frame.sample_rate,
                sample_width=audio_frame.format.bytes,
                channels=len(audio_frame.layout.channels)
            )
            
            # Convert to recognizable format
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            
            # Recognize speech
            with sr.AudioFile(wav_io) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                return text
        except Exception as e:
            return f"Error processing audio: {str(e)}"
    
    def text_to_speech(self, text: str, lang: str = "en") -> bytes:
        """Convert text to speech using gTTS"""
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.read()
        except Exception as e:
            st.error(f"Text-to-speech error: {str(e)}")
            return b""
        
# WebRTC Audio Processor
class AudioProcessor:
    def __init__(self):
        self.voice_handler = VoiceHandler()
        self.audio_buffer = []
        
    def recv(self, frame):
        """Process incoming audio frame"""
        try:
            # Store audio frame
            self.audio_buffer.append(frame)
            
            # Process if buffer has enough frames
            if len(self.audio_buffer) >= 10:  # Process every 10 frames
                # Combine frames and process
                combined_audio = self._combine_frames(self.audio_buffer)
                text = self.voice_handler.process_audio_frame(combined_audio)
                
                # Store recognized text
                if text and 'webrtc_text' not in st.session_state:
                    st.session_state.webrtc_text = text
                
                # Clear buffer
                self.audio_buffer = []
        except Exception as e:
            st.error(f"Audio processing error: {str(e)}")
        
        return frame
    
    def _combine_frames(self, frames):
        """Combine multiple audio frames"""
        # Simple implementation - return first frame for now
        return frames[0] if frames else None