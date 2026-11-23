"""SkinSense AI - Voice Agent
Transcribes patient voice recordings (Hindi or English).
Supports Groq Whisper API, local SpeechRecognition fallback,
and provides transparent metadata so the patient can inspect/correct their transcript.
"""

import os
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()


class VoiceAgent:
    """Agent that transcribes patient audio in English and Hindi."""

    def __init__(self):
        self.groq_api_key = os.environ.get("GROQ_API_KEY")

    def transcribe(self, audio_filepath: str, language: str = "auto") -> Dict[str, Any]:
        """Transcribes audio file with multi-tier fallback."""
        if not audio_filepath or not Path(audio_filepath).exists():
            return {
                "success": False,
                "transcript": "",
                "method": "none",
                "error": "Audio file not found or empty.",
            }

        # Strategy 1: Groq Whisper API (High Accuracy Hindi & English)
        if self.groq_api_key:
            try:
                from groq import Groq
                client = Groq(api_key=self.groq_api_key)
                with open(audio_filepath, "rb") as af:
                    kwargs = {
                        "file": af,
                        "model": os.environ.get("WHISPER_MODEL", "whisper-large-v3"),
                    }
                    if language and language != "auto":
                        kwargs["language"] = language

                    transcription = client.audio.transcriptions.create(**kwargs)
                    return {
                        "success": True,
                        "transcript": transcription.text.strip(),
                        "method": "groq_whisper",
                        "editable": True,
                    }
            except Exception as e:
                print(f"Groq Whisper transcription failed: {e}. Trying local fallback...")

        # Strategy 2: speech_recognition (Google Web Speech API free fallback)
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            from pydub import AudioSegment

            # Convert to wav
            sound = AudioSegment.from_file(audio_filepath)
            wav_path = Path(audio_filepath).with_suffix(".temp.wav")
            sound.export(str(wav_path), format="wav")

            with sr.AudioFile(str(wav_path)) as source:
                audio_data = r.record(source)

            # Try Hindi or English
            try:
                text = r.recognize_google(audio_data, language="hi-IN" if language == "hi" else "en-US")
            except sr.UnknownValueError:
                text = r.recognize_google(audio_data)

            if wav_path.exists():
                wav_path.unlink()

            return {
                "success": True,
                "transcript": text.strip(),
                "method": "google_speech_rec",
                "editable": True,
            }
        except Exception as e:
            # Fallback message
            return {
                "success": False,
                "transcript": "",
                "method": "manual_entry_recommended",
                "error": f"Speech-to-text service unavailable ({e}). Please type your symptoms in the text box.",
            }

