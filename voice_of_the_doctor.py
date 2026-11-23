"""SkinSense AI - Doctor Voice Generation (TTS)
Generates audio guidance in English or Hindi.
Supports Deepgram Aura (high quality) with seamless automatic fallback
to gTTS (Google Text-to-Speech) so it NEVER crashes when offline or without API keys.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOCTOR_AUDIO = BASE_DIR / "doctor_response.mp3"


def convert_text_to_doctor_audio(
    text: str,
    output_filepath: Optional[Path] = None,
    language: str = "en",
) -> Path:
    """Converts doctor response text to MP3 audio using Deepgram or gTTS fallback."""
    if output_filepath is None:
        output_filepath = DEFAULT_DOCTOR_AUDIO

    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    clean_text = text.replace("*", "").replace("#", "").replace("`", "").strip()
    if not clean_text:
        clean_text = "Analysis complete. Please review the clinician report."

    # Strategy 1: Deepgram (if API key available)
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")
    if deepgram_api_key:
        try:
            from deepgram import DeepgramClient
            deepgram = DeepgramClient(api_key=deepgram_api_key)
            audio = deepgram.speak.v1.audio.generate(
                text=clean_text,
                model=os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en"),
                encoding="mp3",
            )
            with output_filepath.open("wb") as file:
                for chunk in audio:
                    file.write(chunk)
            return output_filepath
        except Exception as e:
            print(f"Deepgram TTS failed: {e}. Falling back to gTTS...")

    # Strategy 2: gTTS (Free, reliable, no API key needed, supports English & Hindi)
    try:
        from gtts import gTTS
        lang_code = "hi" if "hi" in language.lower() else "en"
        tts = gTTS(text=clean_text, lang=lang_code, slow=False)
        tts.save(str(output_filepath))
        return output_filepath
    except Exception as e:
        print(f"gTTS audio generation error: {e}")
        # Create silent dummy MP3 if TTS entirely unavailable
        with output_filepath.open("wb") as f:
            f.write(b"")
        return output_filepath
