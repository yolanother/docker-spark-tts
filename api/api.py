"""
requirements.txt:
fastapi
uvicorn
torch
torchaudio
pydub
soundfile
transformers
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import torch
import torchaudio
import os
import uvicorn
import re
import soundfile as sf
from pydub import AudioSegment
from cli.SparkTTS import SparkTTS

# Initialize FastAPI app
app = FastAPI(title="Spark-TTS API", openapi_url="/api/docs/openapi.json", docs_url="/api/docs")

# Load the model
model_dir = "/Spark-TTS/pretrained_models/Spark-TTS-0.5B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SparkTTS(model_dir, device)

# Ensure output directories exist
OUTPUT_DIR = "/data/generated_audio"
SAMPLES_DIR = "/data/samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

def sanitize_filename(filename: str) -> str:
    """
    Remove invalid characters from filenames to prevent path traversal or injection attacks.
    """
    return re.sub(r'[^a-zA-Z0-9_-]', '_', filename)

def convert_to_wav(input_path: str, output_path: str):
    """
    Convert audio file to WAV format.
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(output_path, format="wav")

@app.post("/api/upload-voice-sample/")
async def upload_voice_sample(voice_name: str, audio_file: UploadFile = File(...)):
    """
    Upload a named voice audio file to be used for speaker embedding.
    Supports MP3, M4A, and WAV formats.
    """
    try:
        voice_name = sanitize_filename(voice_name)
        temp_path = f"{SAMPLES_DIR}/{voice_name}.{audio_file.filename.split('.')[-1]}"
        output_path = f"{SAMPLES_DIR}/{voice_name}.wav"
        
        with open(temp_path, "wb") as buffer:
            buffer.write(await audio_file.read())
        
        # Convert to WAV if needed
        if not temp_path.endswith(".wav"):
            convert_to_wav(temp_path, output_path)
            os.remove(temp_path)
        else:
            os.rename(temp_path, output_path)
        
        return {"message": "Voice uploaded and converted successfully", "voice_name": voice_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/generate-audio/")
async def generate_audio(
    text: str,
    voice_name: str = None,
    gender: str = None,
    pitch: str = None,
    speed: str = None,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    audio_file: UploadFile = File(None)
):
    """
    Generate audio from text using a provided voice or uploaded audio for speaker embedding.
    
    Parameters:
    - text: The text to convert to speech
    - voice_name: Name of a previously uploaded voice sample
    - gender: 'male' or 'female' (for voice creation mode)
    - pitch: 'very_low', 'low', 'moderate', 'high', 'very_high' (for voice creation mode)
    - speed: 'very_low', 'low', 'moderate', 'high', 'very_high' (for voice creation mode)
    - temperature: Sampling temperature for controlling randomness (default: 0.8)
    - top_k: Top-k sampling parameter (default: 50)
    - top_p: Top-p (nucleus) sampling parameter (default: 0.95)
    - audio_file: Upload a voice sample for one-time use
    
    Returns a generated audio file.
    """
    try:
        prompt_speech_path = None
        prompt_text = None
        
        # Voice cloning mode (using reference audio)
        if voice_name or audio_file:
            if voice_name:
                voice_name = sanitize_filename(voice_name)
                prompt_speech_path = f"{SAMPLES_DIR}/{voice_name}.wav"
                if not os.path.exists(prompt_speech_path):
                    raise HTTPException(status_code=404, detail="Voice not found")
            elif audio_file:
                temp_path = f"{OUTPUT_DIR}/{audio_file.filename}"
                prompt_speech_path = f"{OUTPUT_DIR}/converted.wav"
                
                with open(temp_path, "wb") as buffer:
                    buffer.write(await audio_file.read())
                
                # Convert to WAV if necessary
                if not temp_path.endswith(".wav"):
                    convert_to_wav(temp_path, prompt_speech_path)
                    os.remove(temp_path)
                else:
                    os.rename(temp_path, prompt_speech_path)
        # Voice creation mode (using parameters)
        elif gender:
            if gender not in ["male", "female"]:
                raise HTTPException(status_code=400, detail="Gender must be 'male' or 'female'")
            if pitch not in ["very_low", "low", "moderate", "high", "very_high"]:
                raise HTTPException(status_code=400, detail="Pitch must be one of: 'very_low', 'low', 'moderate', 'high', 'very_high'")
            if speed not in ["very_low", "low", "moderate", "high", "very_high"]:
                raise HTTPException(status_code=400, detail="Speed must be one of: 'very_low', 'low', 'moderate', 'high', 'very_high'")
        else:
            raise HTTPException(status_code=400, detail="Either voice_name, audio_file, or gender parameters must be provided")

        # Generate audio
        with torch.no_grad():
            wav = model.inference(
                text=text,
                prompt_speech_path=prompt_speech_path,
                prompt_text=prompt_text,
                gender=gender,
                pitch=pitch,
                speed=speed,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

        # Save generated audio
        output_audio_path = f"{OUTPUT_DIR}/output.wav"
        sf.write(output_audio_path, wav, samplerate=16000)

        return FileResponse(output_audio_path, media_type="audio/wav", filename="output.wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/list-voices/")
async def list_voices():
    """
    List all available voices that can be used for speaker embedding.
    Returns a list of voice names without file extensions.
    """
    try:
        # Get all WAV files in the voices directory
        voices = []
        if os.path.exists(SAMPLES_DIR):
            for file in os.listdir(SAMPLES_DIR):
                if file.endswith(".wav"):
                    # Remove the .wav extension to get the voice name
                    voice_name = os.path.splitext(file)[0]
                    voices.append(voice_name)
        
        return {"voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7311)
