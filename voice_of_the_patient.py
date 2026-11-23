"""SkinSense AI - Patient Voice Input & Transcription Module
Records microphone audio and transcribes patient speech using Groq Whisper
or local SpeechRecognition fallback, supporting English and Hindi.
"""

from io import BytesIO
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from pydub import AudioSegment
import speech_recognition as sr

from agents.voice_agent import VoiceAgent

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_voice_agent = VoiceAgent()


def record_audio(file_path: str, timeout: int = 20, phrase_time_limit: int = 15):
    """Records microphone audio and saves as an MP3 file."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        logging.info("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        logging.info("Recording patient audio...")
        audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        logging.info("Recording complete.")

        wav_data = audio_data.get_wav_data()
        audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
        audio_segment.export(file_path, format="mp3", bitrate="128k")
        logging.info(f"Audio exported to {file_path}")


def transcribe_patient_voice(audio_filepath: str, language: str = "auto") -> str:
    """Transcribes patient audio in English or Hindi using VoiceAgent."""
    res = _voice_agent.transcribe(audio_filepath, language=language)
    if res.get("success"):
        return res.get("transcript", "")
    return res.get("error", "Unable to transcribe audio. Please type your symptoms manually.")
