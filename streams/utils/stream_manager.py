import threading
import requests
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

class StreamManager:
    def __init__(self):
        self.streams = {}

    def add_stream(self, name, url):
        if name not in self.streams:
            print(f"Starting stream for {name}")
            stream_thread = threading.Thread(target=self.process_stream, args=(name, url))
            stream_thread.start()
            self.streams[name] = stream_thread

    def process_stream(self, name, url):
        print(f"Processing stream for {name} from {url}")
        response = requests.get(url, stream=True)
        audio_data = BytesIO()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                audio_data.write(chunk)
                self.handle_audio_chunk(name, audio_data.getvalue())
                audio_data.seek(0)
                audio_data.truncate()

    def handle_audio_chunk(self, name, chunk):
        # Convert raw audio chunk to AudioSegment
        audio_segment = AudioSegment.from_raw(BytesIO(chunk), sample_width=2, frame_rate=44100, channels=1)
        
        # Save to a temporary file for speech recognition
        audio_file = BytesIO()
        audio_segment.export(audio_file, format='wav')
        audio_file.seek(0)
        
        # Recognize the audio
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
                print(f"Transcribed text for {name}: {text}")
            except sr.UnknownValueError:
                print(f"Could not understand audio from {name}")
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")
