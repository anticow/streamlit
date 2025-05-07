# Setup Guide for Local Audio Processing App

## Prerequisites

1. **Python 3.11**
   - Required for compatibility with all dependencies

2. **NVIDIA GPU with CUDA support**
   - For optimal performance with Whisper transcription

3. **Ollama**
   - Local LLM server for text summarization

## Installation Steps

### 1. Create a Python environment

```bash
# Create a new Python 3.11 virtual environment
python3.11 -m venv audioai-env

# Activate the environment
# On Windows:
audioai-env\Scripts\activate
# On macOS/Linux:
source audioai-env/bin/activate
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```
I have had to remove torch and torchaudio, then manually reinstall torch+cuda128 and torchaudio+cuda128

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install CUDA (if not already installed)

- Download and install the appropriate CUDA version for your system from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
- Ensure your PyTorch installation is CUDA-compatible

### 4. Install and run Ollama

#### For macOS and Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### For Windows:
- Download the installer from [Ollama's website](https://ollama.com/download)

#### Pull the LLM model:
```bash
# Start the Ollama server
ollama serve

# In a new terminal, pull your preferred model
ollama pull llama3  # Or another model of your choice
```

## Running the Application

1. Save the Python script as `run_strealit_app.py`

2. Run the application:
```bash
python run_strealit_app.py
```

## Troubleshooting

1. **Microphone access issues**
   - Ensure your microphone is properly connected and has system permissions

2. **CUDA errors**
   - Verify compatible versions of PyTorch and CUDA
   - Check GPU drivers are up to date

3. **Ollama connection issues**
   - Ensure Ollama is running with `ollama serve`
   - Check the default endpoint is correct: http://localhost:11434/api/generate
   - Verify the model is pulled correctly with `ollama list`

4. **Memory issues with larger models**
   - For Whisper: Try a smaller model size
   - For Ollama: Use a more efficient model or increase available GPU memory