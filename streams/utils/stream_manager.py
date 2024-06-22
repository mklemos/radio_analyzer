import threading
import subprocess
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os

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
        buffer_size = 500000  # Increase buffer size to capture more audio data
        command = [
            'ffmpeg',
            '-i', url,
            '-f', 'wav',
            '-ar', '16000',
            '-ac', '1',
            '-vn',
            '-'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        buffer = BytesIO()

        while True:
            data = process.stdout.read(4096)  # Read more data at a time
            if not data:
                break
            buffer.write(data)
            if buffer.tell() >= buffer_size:
                buffer.seek(0)
                audio_segment = AudioSegment.from_file(buffer, format='wav')
                audio_file = BytesIO()
                audio_segment.export(audio_file, format='wav')
                audio_file.seek(0)
                print(f"Buffered audio duration for {name}: {len(audio_segment) / 1000} seconds")  # Print audio duration
                self.handle_audio_chunk(name, audio_file)
                buffer.seek(0)
                buffer.truncate()

    def handle_audio_chunk(self, name, audio_file):
        # Save to a temporary file for debugging
        tmp_filename = f'/tmp/{name}.wav'
        with open(tmp_filename, 'wb') as f:
            f.write(audio_file.getbuffer())

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            print(f"Audio length for {name}: {len(audio.get_wav_data()) / 16000} seconds")  # Print audio length
            try:
                text = recognizer.recognize_google(audio)
                print(f"Transcribed text for {name}: {text}")
            except sr.UnknownValueError:
                print(f"Could not understand audio from {name}")
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")

        # Clean up temporary file
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

# Add cleanup code for temporary files if needed
def cleanup_tmp_files():
    tmp_files = [f for f in os.listdir('/tmp') if f.endswith('.wav')]
    for file in tmp_files:
        os.remove(os.path.join('/tmp', file))
