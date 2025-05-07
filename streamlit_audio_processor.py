"""
Streamlit Audio Processing and Note-Taking App
- Records audio from microphone
- Transcribes speech using OpenAI's Whisper model (locally)
- Summarizes content using Ollama
- Maintains conversation history
- Provides a web interface with Streamlit
"""

import os
import sys
import time
import json
import tempfile
import threading
import asyncio
from queue import Queue
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import requests
import streamlit as st
from datetime import datetime
from io import BytesIO

# Import PyTorch and Whisper after Streamlit has initialized
try:
    import torch
    import whisper
except Exception as e:
    st.error(f"Error importing PyTorch or Whisper: {str(e)}")
    st.warning("The application may not function correctly without these libraries.")
    # Create dummy objects to prevent errors
    class DummyTorch:
        def cuda(self):
            return False
        def is_available(self):
            return False
    torch = DummyTorch()
    whisper = None

# Configuration
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_RECORDING_DURATION = 10  # seconds
TEMP_DIR = tempfile.gettempdir()

# Helper function to safely check if asyncio loop is running
def is_asyncio_loop_running():
    """Safely check if an asyncio event loop is running in the current thread."""
    try:
        loop = asyncio.get_running_loop()
        return loop.is_running()
    except RuntimeError:
        # No running event loop
        return False

# Monkey patch to handle asyncio.get_running_loop().is_running() calls
original_get_running_loop = asyncio.get_running_loop

def safe_get_running_loop():
    """A safer version of asyncio.get_running_loop that doesn't raise exceptions."""
    try:
        return original_get_running_loop()
    except RuntimeError:
        # Create and return a new event loop if one doesn't exist
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Apply the monkey patch
asyncio.get_running_loop = safe_get_running_loop

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'device' not in st.session_state:
    st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = ""
if 'stop_recording' not in st.session_state:
    st.session_state.stop_recording = False
if 'recording_thread' not in st.session_state:
    st.session_state.recording_thread = None
if 'data_queue' not in st.session_state:
    st.session_state.data_queue = None
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()

class AudioProcessor:
    @staticmethod
    def load_whisper_model(model_size):
        """Load the Whisper model with the specified size."""
        try:
            if st.session_state.whisper_model is None:
                with st.spinner(f"Loading Whisper model {model_size}..."):
                    st.session_state.whisper_model = whisper.load_model(
                        model_size, 
                        device=st.session_state.device
                    )
                st.success(f"Whisper model {model_size} loaded successfully!")
            return st.session_state.whisper_model
        except Exception as e:
            st.error(f"Error loading Whisper model: {str(e)}")
            st.session_state.processing = False
            if "CUDA" in str(e):
                st.error("CUDA error detected. Falling back to CPU.")
                st.session_state.device = "cpu"
                with st.spinner(f"Retrying: Loading Whisper model {model_size} on CPU..."):
                    st.session_state.whisper_model = whisper.load_model(model_size, device="cpu")
                st.success(f"Whisper model {model_size} loaded on CPU!")
                return st.session_state.whisper_model
            return None
    
    @staticmethod
    def record_audio_buffer(sample_rate, channels, stop_event, data_queue, status_queue):
        """
        Captures audio data from the user's microphone and adds it to a queue for further processing.
        Uses a buffering system for better streaming performance.

        Args:
            sample_rate (int): The sample rate to use for recording.
            channels (int): The number of audio channels to record.
            stop_event (threading.Event): Event to signal when to stop recording.
            data_queue (Queue): Queue to store audio data.
            status_queue (Queue): Queue to store status updates.

        Returns:
            None
        """
        def callback(indata, frames, time, status):
            # Don't access any Streamlit objects here
            data_queue.put(bytes(indata))
            
            # Store status if there's an error
            if status:
                status_queue.put(("error", f"Audio recording error: {status}"))

        try:
            with sd.RawInputStream(
                samplerate=sample_rate,
                dtype="int16",
                channels=channels,
                callback=callback
            ):
                # Just record until the stop event is set
                start_time = time.time()
                while not stop_event.is_set():
                    elapsed = time.time() - start_time
                    # Send elapsed time through the queue
                    status_queue.put(("elapsed", elapsed))
                    time.sleep(0.1)
        except Exception as e:
            # Send error through the queue
            status_queue.put(("error", str(e)))
    
    @staticmethod
    def start_buffered_recording(sample_rate, channels):
        """Start recording audio using a queue-based buffer system."""
        try:
            # Initialize recording components
            data_queue = Queue()
            status_queue = Queue()
            stop_event = threading.Event()
            
            st.session_state.data_queue = data_queue
            st.session_state.status_queue = status_queue
            st.session_state.stop_event = stop_event
            st.session_state.recording = True
            st.session_state.recording_elapsed = 0
            st.session_state.recording_error = None
            st.session_state.audio_status = None
            
            # Create UI elements for recording status
            status_text = st.empty()
            progress_bar = st.progress(0)
            status_text.text("Recording started. Press 'Stop Recording' when finished.")
            
            # Start recording in a separate thread
            st.session_state.recording_thread = threading.Thread(
                target=AudioProcessor.record_audio_buffer,
                args=(sample_rate, channels, stop_event, data_queue, status_queue)
            )
            st.session_state.recording_thread.start()
            
            return status_text, progress_bar
            
        except Exception as e:
            st.error(f"Error starting recording: {str(e)}")
            st.session_state.recording = False
            return None, None
    
    @staticmethod
    def stop_buffered_recording(sample_rate, channels, status_text, progress_bar):
        """Stop the recording and process the recorded audio."""
        try:
            if st.session_state.recording and st.session_state.recording_thread:
                # Signal the recording thread to stop
                st.session_state.stop_event.set()
                st.session_state.recording_thread.join()
                
                # Process the recorded audio
                status_text.text("Processing recorded audio...")
                
                # Combine all audio data from the queue
                audio_data = b"".join(list(st.session_state.data_queue.queue))
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Reshape if needed for multiple channels
                if channels > 1:
                    audio_np = audio_np.reshape(-1, channels)
                
                # Save to a temporary file for playback in Streamlit
                temp_file_path = os.path.join(TEMP_DIR, f"recorded_audio_{int(time.time())}.wav")
                sf.write(temp_file_path, audio_np, sample_rate)
                
                # Update session state
                st.session_state.audio_data = audio_np
                st.session_state.temp_file_path = temp_file_path
                st.session_state.recording = False
                
                # Clear UI elements
                status_text.text("Recording complete!")
                progress_bar.progress(1.0)
                
                return temp_file_path, audio_np
            else:
                st.warning("No active recording to stop.")
                return None, None
                
        except Exception as e:
            st.error(f"Error stopping recording: {str(e)}")
            st.session_state.recording = False
            return None, None
    
    @staticmethod
    def record_audio(duration, sample_rate, channels):
        """Record audio from the microphone for the specified duration (legacy method)."""
        try:
            st.session_state.recording = True
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(f"Recording for {duration} seconds...")
            
            # Check for audio devices
            devices = sd.query_devices()
            if not devices:
                raise RuntimeError("No audio devices found")
            
            # Find a suitable input device
            input_device = None
            for device in devices:
                if device['max_input_channels'] > 0:
                    input_device = device['index']
                    break
            
            if input_device is None:
                raise RuntimeError("No input devices with microphones found")
                
            # Prepare to record audio
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype='float32',
                device=input_device
            )
            
            # Update progress bar during recording
            for i in range(duration):
                time.sleep(1)
                progress_bar.progress((i + 1) / duration)
                status_text.text(f"Recording: {i + 1}/{duration} seconds")
            
            # Wait for recording to complete
            sd.wait()
            status_text.text("Recording complete!")
            
            # Save the recorded audio to a temporary file
            temp_file_path = os.path.join(TEMP_DIR, f"recorded_audio_{int(time.time())}.wav")
            sf.write(temp_file_path, audio_data, sample_rate)
            
            st.session_state.audio_data = audio_data
            st.session_state.temp_file_path = temp_file_path
            st.session_state.recording = False
            
            return temp_file_path
            
        except Exception as e:
            st.error(f"Error during audio recording: {str(e)}")
            if "No input devices with microphones found" in str(e):
                st.error("No microphone detected. Please connect a microphone and try again.")
            
            st.session_state.recording = False
            st.session_state.processing = False
            
            # Create a dummy audio file for testing when no microphone is available
            temp_file_path = os.path.join(TEMP_DIR, f"dummy_audio_{int(time.time())}.wav")
            dummy_duration = 3  # seconds
            dummy_audio = np.zeros((int(sample_rate * dummy_duration), channels), dtype=np.float32)
            sf.write(temp_file_path, dummy_audio, sample_rate)
            
            st.warning("Created dummy audio file for testing purposes since recording failed.")
            return temp_file_path
    
    @staticmethod
    def transcribe_audio(audio_file_path, model):
        """Transcribe the recorded audio using Whisper."""
        try:
            with st.spinner("Transcribing audio..."):
                # Check if the file exists
                if not os.path.exists(audio_file_path):
                    st.error(f"Audio file not found: {audio_file_path}")
                    st.info(f"Current working directory: {os.getcwd()}")
                    st.info(f"Checking if file exists with absolute path: {os.path.abspath(audio_file_path)}")
                    return ""
                
                # Log file details for debugging
                st.info(f"Transcribing file: {audio_file_path} (Size: {os.path.getsize(audio_file_path)} bytes)")
                
                # Load the audio file directly using soundfile instead of relying on Whisper's loader
                # which might be trying to use ffmpeg
                try:
                    # Load audio using soundfile
                    audio_data, sample_rate = sf.read(audio_file_path)
                    
                    # Convert to mono if needed
                    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                        audio_data = audio_data.mean(axis=1)
                    
                    # Resample to 16kHz if needed (Whisper expects 16kHz audio)
                    if sample_rate != 16000:
                        st.warning(f"Resampling audio from {sample_rate}Hz to 16000Hz")
                        # Simple resampling - for better quality, consider using librosa
                        target_length = int(len(audio_data) * 16000 / sample_rate)
                        audio_data = np.interp(
                            np.linspace(0, len(audio_data), target_length),
                            np.arange(len(audio_data)),
                            audio_data
                        )
                    
                    # Normalize audio
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    
                    if np.abs(audio_data).max() > 1.0:
                        audio_data = audio_data / np.abs(audio_data).max()
                    
                    # Transcribe the audio data directly
                    result = model.transcribe(audio_data)
                    transcript = result["text"]
                    
                    st.success("Audio transcribed successfully!")
                except Exception as audio_error:
                    st.error(f"Error processing audio with soundfile: {str(audio_error)}")
                    st.warning("Falling back to Whisper's default audio loading...")
                    
                    # Try with absolute path as fallback
                    abs_path = os.path.abspath(audio_file_path)
                    result = model.transcribe(abs_path)
                    transcript = result["text"]
            
            st.session_state.transcript = transcript
            return transcript
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
            # Add more detailed error information
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            
            # Check if ffmpeg is installed
            try:
                import subprocess
                subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
                st.info("ffmpeg is installed and accessible.")
            except Exception as ffmpeg_error:
                st.error(f"ffmpeg check failed: {str(ffmpeg_error)}")
                st.warning("Whisper may require ffmpeg to be installed and in your PATH for audio processing.")
                st.info("You can download ffmpeg from https://ffmpeg.org/download.html")
            
            st.session_state.transcript = ""
            return ""
    
    @staticmethod
    def summarize_text(transcript, ollama_model, ollama_endpoint, system_prompt):
        """Summarize the transcript using Ollama."""
        if not transcript:
            st.error("No transcript available for summarization.")
            return ""
        
        with st.spinner(f"Summarizing with Ollama ({ollama_model})..."):
            # Prepare the prompt for summarization with note-taking instructions
            prompt = f"""You are an intelligent note-taking and summarization assistant.
            
You should structure your response in the following way:
1. SUMMARY: A concise executive summary (2-3 sentences)
2. KEY POINTS: The main ideas or arguments (3-5 bullet points)
3. ACTION ITEMS: Any tasks, follow-ups, or action items mentioned (if applicable)
4. QUESTIONS: Any questions raised that need answers (if applicable)

Here is the transcript to analyze:

{transcript}"""
            
            # Prepare the request payload with system prompt if provided
            payload = {
                "model": ollama_model,
                "prompt": prompt,
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            # Send the request to Ollama
            try:
                response = requests.post(ollama_endpoint, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    summary = result.get('response', '')
                    st.session_state.summary = summary
                    
                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'transcript': transcript,
                        'summary': summary
                    })
                    
                    return summary
                else:
                    st.error(f"Error from Ollama API: {response.status_code}")
                    st.error(f"Response: {response.text}")
                    return ""
            except Exception as e:
                st.error(f"Failed to communicate with Ollama: {str(e)}")
                st.warning("Is Ollama running? Start it with: ollama serve")
                return ""
    
    @staticmethod
    def export_notes(format_type="markdown"):
        """Export all notes in the specified format."""
        if not st.session_state.conversation_history:
            st.error("No conversation history to export.")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        if format_type == "markdown":
            content = f"# Meeting Notes - {datetime.now().strftime('%Y-%m-%d')}\n\n"
            
            for idx, entry in enumerate(st.session_state.conversation_history):
                content += f"## Session {idx+1} - {entry['timestamp']}\n\n"
                content += "### Summary\n\n"
                content += f"{entry['summary']}\n\n"
                content += "### Full Transcript\n\n"
                content += f"{entry['transcript']}\n\n"
                content += "---\n\n"
            
            filename = f"meeting_notes_{timestamp}.md"
            return filename, content
        
        elif format_type == "json":
            content = json.dumps(st.session_state.conversation_history, indent=2)
            filename = f"meeting_notes_{timestamp}.json"
            return filename, content
        
        elif format_type == "csv":
            df = pd.DataFrame(st.session_state.conversation_history)
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            content = buffer.getvalue().decode()
            filename = f"meeting_notes_{timestamp}.csv"
            return filename, content
        
        else:
            st.error(f"Unsupported export format: {format_type}")
            return None


def start_buffered_recording_session():
    """Start a new recording session using the buffered recording approach."""
    if not st.session_state.recording and not st.session_state.processing:
        try:
            st.session_state.processing = True
            
            # Get configuration from session state
            sample_rate = st.session_state.sample_rate
            channels = st.session_state.channels
            whisper_model_size = st.session_state.whisper_model_size
            
            # Load Whisper model if not already loaded or if model size changed
            if whisper_model_size != st.session_state.get('loaded_model_size'):
                st.session_state.whisper_model = None
                st.session_state.loaded_model_size = whisper_model_size
            
            model = AudioProcessor.load_whisper_model(whisper_model_size)
            
            # Check if model loaded successfully
            if model is None:
                st.error("Failed to load Whisper model. Please check the logs and try again.")
                st.session_state.processing = False
                return
            
            # Start buffered recording
            status_text, progress_bar = AudioProcessor.start_buffered_recording(sample_rate, channels)
            
            # Store UI elements in session state for later use
            st.session_state.status_text = status_text
            st.session_state.progress_bar = progress_bar
            
            st.session_state.processing = False
            
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.session_state.processing = False


def stop_buffered_recording_session():
    """Stop the current recording session and process the audio."""
    if st.session_state.recording:
        try:
            st.session_state.processing = True
            
            # Get configuration from session state
            sample_rate = st.session_state.sample_rate
            channels = st.session_state.channels
            whisper_model_size = st.session_state.whisper_model_size
            ollama_model = st.session_state.ollama_model
            ollama_endpoint = st.session_state.ollama_endpoint
            system_prompt = st.session_state.system_prompt
            
            # Stop the recording
            audio_file_path, audio_np = AudioProcessor.stop_buffered_recording(
                sample_rate, 
                channels, 
                st.session_state.status_text, 
                st.session_state.progress_bar
            )
            
            if audio_file_path and audio_np is not None and audio_np.size > 0:
                # Display audio player
                st.audio(audio_file_path)
                
                # Get the model
                model = st.session_state.whisper_model
                
                # Transcribe
                transcript = AudioProcessor.transcribe_audio(audio_file_path, model)
                
                # Summarize
                summary = AudioProcessor.summarize_text(transcript, ollama_model, ollama_endpoint, system_prompt)
            else:
                st.warning("No audio data was recorded or recording was too short.")
            
            st.session_state.processing = False
            
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.session_state.processing = False


def start_recording():
    """Start recording audio using the legacy fixed-duration approach."""
    if not st.session_state.recording and not st.session_state.processing:
        try:
            st.session_state.processing = True
            
            # Get configuration from session state
            sample_rate = st.session_state.sample_rate
            channels = st.session_state.channels
            duration = st.session_state.recording_duration
            whisper_model_size = st.session_state.whisper_model_size
            ollama_model = st.session_state.ollama_model
            ollama_endpoint = st.session_state.ollama_endpoint
            system_prompt = st.session_state.system_prompt
            
            # Load Whisper model if not already loaded or if model size changed
            if whisper_model_size != st.session_state.get('loaded_model_size'):
                st.session_state.whisper_model = None
                st.session_state.loaded_model_size = whisper_model_size
            
            model = AudioProcessor.load_whisper_model(whisper_model_size)
            
            # Check if model loaded successfully
            if model is None:
                st.error("Failed to load Whisper model. Please check the logs and try again.")
                st.session_state.processing = False
                return
            
            # Record audio
            audio_file_path = AudioProcessor.record_audio(duration, sample_rate, channels)
            
            # Display audio player
            st.audio(audio_file_path)
            
            # Transcribe
            transcript = AudioProcessor.transcribe_audio(audio_file_path, model)
            
            # Summarize
            summary = AudioProcessor.summarize_text(transcript, ollama_model, ollama_endpoint, system_prompt)
            
            st.session_state.processing = False
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.session_state.processing = False


def main():
    st.set_page_config(
        page_title="AI Note-Taking Assistant",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("🎙️ AI Note-Taking Assistant")
    st.markdown(
        """
        This application records audio, transcribes it using Whisper, and generates meeting notes using Ollama.
        """
    )
    
    # Process status updates from the recording thread
    if st.session_state.recording and 'status_queue' in st.session_state:
        # Process all available status updates
        status_queue = st.session_state.status_queue
        while not status_queue.empty():
            try:
                status_type, status_value = status_queue.get_nowait()
                
                if status_type == "elapsed":
                    # Update elapsed time
                    st.session_state.recording_elapsed = status_value
                    
                    # Update UI if elements exist
                    if 'progress_bar' in st.session_state:
                        if status_value <= 60:  # Cap at 60 seconds for progress bar
                            st.session_state.progress_bar.progress(min(status_value / 60, 1.0))
                    
                    if 'status_text' in st.session_state:
                        st.session_state.status_text.text(f"Recording... {int(status_value)} seconds")
                
                elif status_type == "error":
                    # Display error
                    st.error(f"Recording error: {status_value}")
            except Exception:
                # If there's an error processing the queue, just continue
                pass
    
    # Initialize session state for model loading
    if 'loaded_model_size' not in st.session_state:
        st.session_state.loaded_model_size = None
    
    # Check for CUDA and display status
    try:
        if torch.cuda.is_available():
            st.session_state.device = "cuda"
            cuda_info = torch.cuda.get_device_name(0)
            st.success(f"✅ CUDA detected - using GPU acceleration ({cuda_info})")
        else:
            st.session_state.device = "cpu"
            st.warning("⚠️ CUDA not available - using CPU (will be slower)")
    except Exception as e:
        st.session_state.device = "cpu"
        st.warning(f"⚠️ Error checking CUDA: {str(e)}. Using CPU instead.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Recording Mode")
        recording_mode = st.radio(
            "Recording Mode",
            ["Buffered (Manual Stop)", "Fixed Duration"],
            index=0,
            help="Buffered mode allows you to record until you manually stop. Fixed duration records for a set time."
        )
        
        st.subheader("Audio Settings")
        st.session_state.sample_rate = st.slider("Sample Rate (Hz)", 8000, 48000, DEFAULT_SAMPLE_RATE, 1000)
        st.session_state.channels = st.selectbox("Audio Channels", [1, 2], index=0)
        
        if recording_mode == "Fixed Duration":
            st.session_state.recording_duration = st.slider(
                "Recording Duration (seconds)", 
                5, 300, DEFAULT_RECORDING_DURATION, 5
            )
        
        st.subheader("AI Models")
        st.session_state.whisper_model_size = st.selectbox(
            "Whisper Model Size", 
            ["tiny", "base", "small", "medium", "large"],
            index=1
        )
        st.session_state.ollama_model = st.text_input("Ollama Model", DEFAULT_OLLAMA_MODEL)
        st.session_state.ollama_endpoint = st.text_input("Ollama API Endpoint", DEFAULT_OLLAMA_ENDPOINT)
        
        st.subheader("Note-Taking Style")
        st.session_state.system_prompt = st.text_area(
            "System Prompt", 
            "You are a professional note-taker and summarization assistant. Your job is to extract key information, organize it clearly, and identify action items.",
            height=150
        )
        
        st.markdown("---")
        
        # Export options
        st.subheader("Export Notes")
        export_format = st.selectbox("Export Format", ["markdown", "json", "csv"], index=0)
        
        if st.button("Export All Notes"):
            result = AudioProcessor.export_notes(export_format)
            if result:
                filename, content = result
                
                # Create download button
                st.download_button(
                    label=f"Download {filename}",
                    data=content,
                    file_name=filename,
                    mime="text/plain"
                )
        
        if st.button("Clear History"):
            if st.session_state.conversation_history:
                st.session_state.conversation_history = []
                st.success("Conversation history cleared!")
    
    # Create two columns for the main interface
    col1, col2 = st.columns([3, 2])
    
    # Left column - Current Session
    with col1:
        st.header("Current Session")
        
        # Recording buttons based on mode
        if recording_mode == "Buffered (Manual Stop)":
            # For buffered mode, show Start/Stop buttons
            col_start, col_stop = st.columns(2)
            
            with col_start:
                if st.session_state.processing or st.session_state.recording:
                    st.button("Recording in progress...", disabled=True, use_container_width=True)
                else:
                    if st.button("Start Recording", use_container_width=True, type="primary"):
                        start_buffered_recording_session()
            
            with col_stop:
                if not st.session_state.recording:
                    st.button("Stop Recording", disabled=True, use_container_width=True)
                else:
                    if st.button("Stop Recording", use_container_width=True, type="secondary"):
                        stop_buffered_recording_session()
        else:
            # For fixed duration mode, show only Start button
            if st.session_state.processing:
                st.button("Processing...", disabled=True, use_container_width=True)
            else:
                if st.button("Start Recording", use_container_width=True, type="primary"):
                    start_recording()
        
        # Display current transcript and summary
        if st.session_state.transcript:
            with st.expander("Transcript", expanded=True):
                st.write(st.session_state.transcript)
        
        if st.session_state.summary:
            with st.expander("AI Notes", expanded=True):
                st.markdown(st.session_state.summary)
    
    # Right column - Conversation History
    with col2:
        st.header("Conversation History")
        
        if not st.session_state.conversation_history:
            st.info("No conversation history yet. Start recording to add entries.")
        else:
            for idx, entry in enumerate(reversed(st.session_state.conversation_history)):
                with st.expander(f"Session {len(st.session_state.conversation_history) - idx}: {entry['timestamp']}", expanded=(idx == 0)):
                    st.markdown("### Summary")
                    st.markdown(entry['summary'])
                    
                    st.markdown("### Transcript")
                    st.markdown(entry['transcript'])


if __name__ == "__main__":
    # Check if we're running directly or through the wrapper script
    if os.environ.get("STREAMLIT_DISABLE_WATCHER") != "true":
        st.warning("""
        ⚠️ For better compatibility with PyTorch, it's recommended to run this app using the wrapper script:
        ```
        python run_streamlit_app.py
        ```
        This helps avoid potential conflicts between PyTorch and Streamlit's module watcher.
        """)
    
    # Ensure we have an asyncio event loop before starting
    try:
        # Check if we're already in an event loop
        if not is_asyncio_loop_running():
            # Create a new event loop if one doesn't exist
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except Exception as e:
        st.error(f"Error setting up asyncio event loop: {str(e)}")
    
    main()
