import threading
import subprocess
import speech_recognition as sr
from pydub import AudioSegment
from pydub.effects import normalize
from io import BytesIO
import os
from streams.models import RadioStation, Transcription

class StreamManager:
    def __init__(self):
        self.streams = {}
        self.transcription_cache = {}

    def add_stream(self, name, url):
        if name not in self.streams:
            print(f"Starting stream for {name}")
            stream_thread = threading.Thread(target=self.process_stream, args=(name, url))
            stream_thread.start()
            self.streams[name] = stream_thread

    def process_stream(self, name, url):
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

        while True:
            retcode = process.poll()
            if retcode is not None:
                break
            try:
                self.handle_audio_chunk(name, process.stdout)
            except Exception as e:
                print(f"Error processing audio chunk for {name}: {e}")

    def handle_audio_chunk(self, name, audio_stream):
        chunk_size = 4096
        buffer = BytesIO()

        while True:
            audio_data = audio_stream.read(chunk_size)
            if not audio_data:
                break
            buffer.write(audio_data)

            # If buffer size is sufficient, process the audio data
            if buffer.tell() > 16000 * 2 * 5:  # 5 seconds of audio at 16kHz, 16-bit mono
                buffer.seek(0)

                # Validate the audio data
                try:
                    audio_segment = AudioSegment.from_file(buffer, format='wav')
                except Exception as e:
                    print(f"Invalid audio file for {name}: {e}")
                    buffer.seek(0)
                    buffer.truncate()
                    continue

                # Normalize audio
                audio_segment = normalize(audio_segment)

                # Convert to BytesIO for speech recognition
                audio_file = BytesIO()
                audio_segment.export(audio_file, format='wav')
                audio_file.seek(0)

                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)
                    print(f"Audio length for {name}: {len(audio.get_wav_data()) / 16000} seconds")  # Print audio length
                    try:
                        text = recognizer.recognize_google(audio)
                        print(f"Transcribed text for {name}: {text}")

                        # Check for duplicates
                        if name not in self.transcription_cache:
                            self.transcription_cache[name] = []
                        if text not in self.transcription_cache[name]:
                            # Save transcription to the database
                            Transcription.objects.create(station=RadioStation.objects.get(name=name), text=text)
                            self.transcription_cache[name].append(text)
                            if len(self.transcription_cache[name]) > 10:  # Limit cache size
                                self.transcription_cache[name].pop(0)
                    except sr.UnknownValueError:
                        print(f"Could not understand audio from {name}")
                    except sr.RequestError as e:
                        print(f"Error with speech recognition service: {e}")

                # Save the temporary file for inspection
                temp_copy = f'/tmp/{name}_copy.wav'
                with open(temp_copy, 'wb') as f:
                    f.write(audio_file.getbuffer())
                print(f"Saved a copy of the audio file to {temp_copy}")

                # Reset the buffer
                buffer.seek(0)
                buffer.truncate()

        # Add cleanup code for temporary files if needed
        def cleanup_tmp_files():
            tmp_files = [f for f in os.listdir('/tmp') if f.endswith('.wav')]
            for file in tmp_files:
                os.remove(os.path.join('/tmp', file))

        cleanup_tmp_files()

