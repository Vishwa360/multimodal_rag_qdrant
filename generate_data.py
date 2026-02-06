from gtts import gTTS
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

text = "Hello, this is a test. We are checking if the speech to text agent works via LangGraph."
tts = gTTS(text)
tts.save("data/sample.mp3")

print("Generated data/sample.mp3")
