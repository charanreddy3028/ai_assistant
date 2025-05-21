import os
from dotenv import load_dotenv
from elevenlabs import save, play
from elevenlabs.client import ElevenLabs

# Load API key from .env
load_dotenv()
api_key = os.getenv("ELEVEN_API_KEY")

if not api_key:
    raise RuntimeError("❌ ELEVEN_API_KEY is missing from the .env file")

# Init ElevenLabs client
client = ElevenLabs(api_key=api_key)

# Text and voice ID
voice_id = "EXAVITQu4vr4xnSDxMaL"
text = "This is a test. Your ElevenLabs voice is working perfectly."

# Generate speech (returns generator)
audio_stream = client.text_to_speech.convert(
    voice_id=voice_id,
    text=text,
    model_id="eleven_monolingual_v1"
)

# Convert generator to bytes
audio_bytes = b"".join(audio_stream)

# Print length for verification
print(f"Audio byte length: {len(audio_bytes)}")

# Save and play
filename = "output_voice.mp3"
save(audio_bytes, filename)
print(f"✅ Audio saved to {filename}")

# Optional: play the audio
play(audio_bytes)
