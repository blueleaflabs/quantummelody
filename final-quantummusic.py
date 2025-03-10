# 1. CONSTANTS & IMPORTS

import os
import sys
import psycopg2
from psycopg2.extras import Json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from matplotlib import colormaps
import scipy.signal  # (if not already imported)
from scipy.signal import find_peaks
from scipy.signal import iirnotch, filtfilt
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator  # Updated import for Qiskit Aer
from qiskit.circuit import ParameterVector
from qiskit.visualization import plot_histogram
import optuna
import concurrent.futures


from scipy.signal import butter, filtfilt

# For Praat-based HNR
import parselmouth
from parselmouth.praat import call

# For audio playback in Jupyter
from IPython.display import Audio, display

# For LUFS-based loudness
try:
    import pyloudnorm as pyln
    LOUDNORM_AVAILABLE = True
except ImportError:
    LOUDNORM_AVAILABLE = False
    print("Warning: pyloudnorm not installed, LUFS computations will be skipped.")

# Quantum Computing imports
# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator  # modern Aer simulator from qiskit-aer


# Directory constants
INPUT_DIR = "data/trainingdata"
OUTPUT_DIR = "output"

# Database constants
DB_NAME = "quantummusic"
DB_HOST = "localhost"
DB_USER = "postgres"  # placeholder
DB_PASSWORD = "postgres"  # placeholder

# Audio processing constants
STANDARD_SR = 44100  # Standard sampling rate
SILENCE_THRESHOLD_DB = -55  # dB threshold for silence trimming

# Band-pass filter constants
LOW_FREQ = 80.0
HIGH_FREQ = 5000.0

# Visualization constants
FIG_SIZE = (10, 4)

# Save-to-DB constant
SAVE_TO_DB = True

# Frame-based approach for pitch detection
FRAME_SIZE = 2048
HOP_LENGTH = 512  # normal frames
# Praat-based chunk size
PRAAT_CHUNK_SIZE = 2048  # for Praat

# Deviation threshold in cents, for dev_flag
DEVIATION_THRESHOLD = 50.0

# Multi-chunk tempo analysis
TEMPO_CHUNK_SIZE_MEDIUM = 4096
TEMPO_CHUNK_SIZE_LARGE = 16384

# New constants for advanced vocal feature extraction
VOCAL_FEATURE_CHUNK_SIZE = 16384       # New chunk size for advanced feature extraction (samples)
VOCAL_FEATURE_CHUNK_HOP = 4096        # Hop size for overlapping chunks (e.g., 50% overlap)

# NEW constant for LUFS calculations (0.5s @ 44.1kHz)
LUFS_CHUNK_SIZE = 22050

# --- NEW CONSTANTS FOR ADVANCED VOCAL FEATURE EXTRACTION ---

# Formant analysis parameters
FORMANT_ANALYSIS_TIME = 0.1            # Time (in seconds) at which to extract formants
FORMANT_TIME_STEP = 0.01               # Time step for formant analysis
MAX_NUMBER_OF_FORMANTS = 5             # Maximum number of formants computed by Praat
MAXIMUM_FORMANT_FREQUENCY = 5500       # Maximum frequency considered for formant analysis
NUM_FORMANTS_TO_EXTRACT = 3            # Number of formants to extract (e.g., F1, F2, F3)

# Pitch / jitter / shimmer parameters
MIN_F0 = 75                          # Minimum expected fundamental frequency (Hz)
MAX_F0 = 500                         # Maximum expected fundamental frequency (Hz)
JITTER_TIME_STEP = 0.0001             # Time step for jitter computation
JITTER_MIN_PERIOD = 0.02              # Minimum period threshold for jitter
JITTER_MAX_PERIOD = 1.3               # Maximum period threshold factor for jitter
SHIMMER_MIN_AMPLITUDE = 0.0001         # Minimum amplitude threshold for shimmer
SHIMMER_MAX_AMPLITUDE = 0.02           # Maximum amplitude threshold for shimmer
SHIMMER_FACTOR = 1.6                 # Shimmer factor (per Praat defaults)

# Vibrato analysis parameters
VIBRATO_MIN_HZ = 3                   # Lower bound for vibrato rate (Hz)
VIBRATO_MAX_HZ = 10                  # Upper bound for vibrato rate (Hz)
MEDIAN_FILTER_KERNEL_SIZE = 9        # Kernel size for median filtering pitch contours


# Ensure output directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_imports():
    print("Imports tested, all modules are available.")

test_imports()



# 2. NOVEL HEURISTICS

def classify_pitch_deviation(dev_cents):
    """
    Classify pitch deviation into categories based on absolute distance from nearest note.
    """
    if dev_cents is None or np.isnan(dev_cents):
        return "unknown"
    dev_abs = abs(dev_cents)
    if dev_abs < 10:
        return "good"
    elif dev_abs < 30:
        return "fair"
    elif dev_abs < 50:
        return "poor"
    elif dev_abs < 100:
        return "very poor"
    else:
        return "completely missing the note"


