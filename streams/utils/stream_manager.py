import threading
import requests

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
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                self.handle_audio_chunk(name, chunk)

    def handle_audio_chunk(self, name, chunk):
        # Process the audio chunk (e.g., convert to text)
        print(f"Handling audio chunk for {name}")
        pass
