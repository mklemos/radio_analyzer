import threading
import subprocess
import speech_recognition as sr
from pydub import AudioSegment
from pydub.effects import normalize
from io import BytesIO
import os
import uuid
import time
from streams.models import RadioStation, Transcription

RIFF_HEADER = b'RIFF'

class StreamManager:
    def __init__(self):
        self.streams = {}
        self.transcription_cache = {}
        self.cleanup_interval = 60  # Cleanup temp files every 60 seconds
        self.start_cleanup_thread()

    def add_stream(self, name, url):
        if name not in self.streams:
            print(f"Starting stream for {name}")
            stream_thread = threading.Thread(target=self.process_stream, args=(name, url))
            stream_thread.start()
            self.streams[name] = stream_thread

    def process_stream(self, name, url):
        while True:
            print(f"Processing stream for {name} from {url}")
            command = [
                'ffmpeg',
                '-i', url,
                '-f', 'wav',
                '-ar', '16000',
                '-ac', '1',
                '-vn',  # No video
                'pipe:1'
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)

            retcode = process.poll()
            if retcode is not None:
                process.terminate()
                time.sleep(5)  # Short delay before restarting the stream process
                continue

            try:
                self.handle_audio_chunk(name, process.stdout)
            except Exception as e:
                print(f"Error processing audio chunk for {name}: {e}")
                process.terminate()
                time.sleep(5)  # Short delay before restarting the stream process
                continue

    def handle_audio_chunk(self, name, audio_stream):
        chunk_size = 16000 * 2 * 10  # Process every 10 seconds of audio
        buffer = audio_stream.read(chunk_size)

        if not buffer:
            return

        print(f"Buffer size: {len(buffer)}")
        print(f"Buffer sample: {buffer[:100]}")

        if not buffer.startswith(RIFF_HEADER):
            print("Detected missing RIFF header. Restarting the stream.")
            raise Exception("Missing RIFF header, restarting stream.")

        unique_id = uuid.uuid4()
        temp_audio_path = f'/tmp/{name}_{unique_id}.wav'

        try:
            audio_segment = AudioSegment.from_file(BytesIO(buffer), format='wav')
            audio_segment = normalize(audio_segment)
            audio_segment.export(temp_audio_path, format='wav')

            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                audio = recognizer.record(source)
                print(f"Audio length for {name}: {len(audio.get_wav_data()) / 16000} seconds")
                try:
                    text = recognizer.recognize_google(audio)
                    print(f"Transcribed text for {name}: {text}")

                    if name not in self.transcription_cache:
                        self.transcription_cache[name] = []
                    if text not in self.transcription_cache[name]:
                        Transcription.objects.create(station=RadioStation.objects.get(name=name), text=text)
                        self.transcription_cache[name].append(text)
                        if len(self.transcription_cache[name]) > 10:
                            self.transcription_cache[name].pop(0)
                except sr.UnknownValueError:
                    print(f"Could not understand audio from {name}")
                except sr.RequestError as e:
                    print(f"Error with speech recognition service: {e}")

            temp_copy = f'/tmp/{name}_{unique_id}_copy.wav'
            os.rename(temp_audio_path, temp_copy)
            print(f"Saved a copy of the audio file to {temp_copy}")

        except Exception as e:
            print(f"Invalid audio file for {name}: {e}")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    def start_cleanup_thread(self):
        cleanup_thread = threading.Thread(target=self.cleanup_temp_files_periodically)
        cleanup_thread.daemon = True  # Ensure the thread exits when the main program does
        cleanup_thread.start()

    def cleanup_tmp_files(self):
        print("Cleaning up temp files...")
        tmp_files = [f for f in os.listdir('/tmp') if f.endswith('.wav')]
        for file in tmp_files:
            file_path = os.path.join('/tmp', file)
            try:
                os.remove(file_path)
                print(f"Deleted temporary file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

    def cleanup_temp_files_periodically(self):
        while True:
            self.cleanup_tmp_files()
            time.sleep(self.cleanup_interval)

# Start StreamManager and add a stream
if __name__ == "__main__":
    manager = StreamManager()
    manager.add_stream("NPR", "https://npr-ice.streamguys1.com/live.mp3")
