"""
Wrapper script to run the Streamlit audio processor application with special handling for PyTorch.
This script sets environment variables to disable Streamlit's module watcher for PyTorch modules.
"""

import os
import sys
import subprocess
import importlib.util

# Set environment variables to disable Streamlit's module watcher
os.environ["STREAMLIT_DISABLE_WATCHER"] = "true"
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"

# Monkey patch torch to avoid the __path__._path issue
# This needs to be done before importing torch
if importlib.util.find_spec("torch") is not None:
    import types
    import sys
    
    # Create a dummy module to replace torch temporarily
    dummy_torch = types.ModuleType("torch")
    dummy_torch.__path__ = []
    
    # Store the original torch module if it's already imported
    original_torch = sys.modules.get("torch")
    
    # Replace torch with our dummy module
    sys.modules["torch"] = dummy_torch

# Get the path to the Streamlit audio processor script
script_dir = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(script_dir, "streamlit_audio_processor.py")

# Run the Streamlit application with the server.fileWatcherType set to none
cmd = [
    sys.executable, 
    "-m", 
    "streamlit", 
    "run", 
    app_path,
    "--server.fileWatcherType=none"
]

print("Starting Streamlit with file watcher disabled...")
print("Command:", " ".join(cmd))
subprocess.run(cmd)

# Restore the original torch module if it existed
if original_torch is not None:
    sys.modules["torch"] = original_torch
