import threading
import subprocess
import speech_recognition as sr
from pydub import AudioSegment
from pydub.effects import normalize
from io import BytesIO
import os
import uuid
import time
import collections
from datetime import datetime
from collections import defaultdict
from streams.models import RadioStation, Transcription, Segment
from transformers import pipeline
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import groupby
from django.utils import timezone

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable to handle tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

RIFF_HEADER = b'RIFF'

class SmartSummarizer:
    def __init__(self, similarity_threshold=0.7, summary_word_threshold=30):
        self.similarity_threshold = similarity_threshold
        self.summary_word_threshold = summary_word_threshold
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.summary_history = []

    def clean_text(self, text):
        # Remove filler words and phrases
        filler_words = ['um', 'uh', 'like', 'you know', 'I mean', 'kind of', 'sort of', 'basically']
        for word in filler_words:
            text = re.sub(r'\b' + word + r'\b', '', text, flags=re.IGNORECASE)
        
        # Remove repeated words and phrases
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def post_process_summary(self, summary):
        # Remove repeated sentences
        sentences = summary.split('.')
        unique_sentences = []
        for sentence in sentences:
            if sentence.strip() and sentence.strip() not in unique_sentences:
                unique_sentences.append(sentence.strip())
        
        # Rejoin sentences
        summary = '. '.join(unique_sentences)
        
        # Remove repeated phrases within sentences
        words = summary.split()
        unique_words = [x[0] for x in groupby(words)]
        summary = ' '.join(unique_words)
        
        return summary

    def extract_key_phrases(self, text):
        # Using NLTK for key phrase extraction
        words = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(words)
        key_phrases = [word for word, tag in tagged if tag.startswith('NN') or tag.startswith('VB')]
        return ' '.join(key_phrases)

    def summarize_text(self, text, max_length=250):
        cleaned_text = self.clean_text(text)
        word_count = len(cleaned_text.split())
        if word_count < self.summary_word_threshold:
            return None  # We'll handle this case in create_segment

        max_length = min(max_length, max(50, word_count // 2))

        new_summary = self.summarizer(cleaned_text, max_length=max_length, min_length=min(50, max_length // 2), do_sample=False, num_beams=4)[0]['summary_text']
        new_summary = self.post_process_summary(new_summary)

        if not self.summary_history:
            self.summary_history.append(new_summary)
            return new_summary

        vectorizer = TfidfVectorizer().fit_transform([new_summary] + self.summary_history)
        cosine_similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

        if any(sim > self.similarity_threshold for sim in cosine_similarities):
            most_similar_index = cosine_similarities.argmax()
            context = self.summary_history[most_similar_index]
            combined_summary = f"{context} {new_summary}"
            combined_summary = self.post_process_summary(combined_summary)
            self.summary_history.append(combined_summary)
            return combined_summary
        else:
            self.summary_history.append(new_summary)
            return new_summary


class StreamManager:
    def __init__(self):
        self.streams = {}
        self.transcription_cache = defaultdict(list)
        self.audio_lock = threading.Lock()
        self.db_lock = threading.Lock()
        self.cleanup_interval = 300  # 5 minutes
        self.start_cleanup_thread()
        self.start_persistence_thread()
        
        self.summarizer = SmartSummarizer()  # Initialize SmartSummarizer instead of pipeline directly
        
        self.accumulated_transcript = ""
        self.word_threshold = 300
        self.max_accumulation_words = 1000
        self.time_threshold = 60  # 60 seconds
        self.last_summary_time = time.time()
        
        self.summary_history = []
        self.max_summary_history = 3
        self.similarity_threshold = 0.3

        self.audio_buffer = collections.deque(maxlen=40)  # 20 minutes of 30-second chunks
        
        self.accumulated_text_for_summary = ""
        self.accumulated_word_count = 0
        self.summary_word_threshold = 100  # Adjust this value as needed

    def add_stream(self, name, url):
        if name not in self.streams:
            logger.info(f"Starting stream for {name}")
            stream_thread = threading.Thread(target=self.process_stream, args=(name, url))
            stream_thread.start()
            self.streams[name] = stream_thread

    def process_stream(self, name, url):
        while True:
            logger.info(f"Processing stream for {name} from {url}")
            command = [
                'ffmpeg',
                '-i', url,
                '-f', 'wav',
                '-ar', '16000',
                '-ac', '1',
                '-vn',
                'pipe:1'
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)

            retcode = process.poll()
            if retcode is not None:
                process.terminate()
                time.sleep(1)
                continue

            try:
                self.handle_audio_chunk(name, process.stdout)
            except Exception as e:
                logger.error(f"Error processing audio chunk for {name}: {e}", exc_info=True)
                process.terminate()
                time.sleep(1)
                continue

    def handle_audio_chunk(self, name, audio_stream):
        chunk_size = 16000 * 2 * 30  # 30 seconds of audio
        buffer = audio_stream.read(chunk_size)

        if not buffer:
            return

        logger.info(f"Buffer size: {len(buffer)}")
        logger.debug(f"Buffer sample: {buffer[:100]}")

        if not buffer.startswith(RIFF_HEADER):
            logger.warning("Detected missing RIFF header. Restarting the stream.")
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
                logger.info(f"Audio length for {name}: {len(audio.get_wav_data()) / 16000} seconds")
                text = recognizer.recognize_google(audio)
                logger.info(f"Transcribed text for {name}: {text}")

            if not text.strip():
                logger.info(f"No text transcribed for {name}. Skipping summarization and topic segmentation.")
                return

            with self.audio_lock:
                self.audio_buffer.append((audio_segment, text))
                self.accumulated_transcript += " " + text
                self.accumulated_transcript = self.summarizer.clean_text(self.accumulated_transcript)
                self.process_buffer(name)

            with self.audio_lock:
                temp_copy = f'/tmp/{name}_{unique_id}_copy.wav'
                os.rename(temp_audio_path, temp_copy)
                logger.info(f"Saved a copy of the audio file to {temp_copy}")

        except sr.UnknownValueError:
            logger.warning(f"Could not understand audio from {name}")
        except sr.RequestError as e:
            logger.error(f"Error with speech recognition service: {e}")
        except Exception as e:
            logger.error(f"Error in handle_audio_chunk for {name}: {e}", exc_info=True)
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    def process_buffer(self, name):
        while len(self.audio_buffer) >= 2:  # Process every minute (2 * 30-second chunks)
            segment = self.create_segment(name)
            if segment:
                self.store_segment(segment)

    def create_segment(self, name):
        audio = AudioSegment.empty()
        text = ""
        with self.audio_lock:
            while self.audio_buffer:
                chunk = self.audio_buffer.popleft()
                audio += chunk[0]
                text += chunk[1] + " "
        
        self.accumulated_text_for_summary += text
        self.accumulated_word_count += len(text.split())
        
        summary = None
        if self.accumulated_word_count >= self.summary_word_threshold:
            summary = self.summarizer.summarize_text(self.accumulated_text_for_summary, max_length=250)
            self.accumulated_text_for_summary = ""
            self.accumulated_word_count = 0
        
        if summary:
            return {
                'station': name,
                'audio': audio,
                'text': text,
                'summary': summary,
                'timestamp': timezone.now()
            }
        else:
            return None

    def store_segment(self, segment):
        if segment is None:
            return
        
        audio_path = f"/tmp/{segment['station']}_{segment['timestamp'].strftime('%Y%m%d%H%M%S')}.wav"
        segment['audio'].export(audio_path, format="wav")
        Segment.objects.create(
            station=RadioStation.objects.get(name=segment['station']),
            timestamp=segment['timestamp'],
            audio_path=audio_path,
            text=segment['text'],
            summary=segment['summary']
        )

    def process_transcript(self, new_transcript, name):
        with self.audio_lock:
            self.accumulated_transcript += " " + new_transcript
            self.accumulated_transcript = self.summarizer.clean_text(self.accumulated_transcript)
        
        word_count = len(self.accumulated_transcript.split())
        current_time = time.time()
        time_since_last_summary = current_time - self.last_summary_time
        
        if word_count >= self.word_threshold or time_since_last_summary >= self.time_threshold:
            summary = self.summarizer.summarize_text(self.accumulated_transcript, max_length=250)
            if summary:
                logger.info(f"Summary for {name}: {summary}")
                
                self.summary_history.append(summary)
                if len(self.summary_history) > self.max_summary_history:
                    self.summary_history.pop(0)
                
                words_to_keep = min(word_count // 2, self.max_accumulation_words // 2)
                self.accumulated_transcript = " ".join(self.accumulated_transcript.split()[-words_to_keep:])
                
                self.last_summary_time = current_time
                
                for attempt in range(3):
                    if self.db_lock.acquire(timeout=5):
                        try:
                            Transcription.objects.create(
                                station=RadioStation.objects.get(name=name),
                                text=new_transcript,
                                summary=summary
                            )
                            self.transcription_cache[name].append(new_transcript)
                            if len(self.transcription_cache[name]) > 10:
                                self.transcription_cache[name].pop(0)
                            break
                        finally:
                            self.db_lock.release()
                    else:
                        logger.warning(f"Failed to acquire lock in process_transcript (attempt {attempt + 1})")
                        time.sleep(1)
                else:
                    logger.error("Failed to acquire lock in process_transcript after 3 attempts")
            elif word_count >= self.max_accumulation_words:
                summary = self.summarizer.summarize_text(self.accumulated_transcript, max_length=300)
                logger.info(f"Forced summary for {name}: {summary}")
                self.accumulated_transcript = ""
                self.last_summary_time = current_time

    def start_cleanup_thread(self):
        cleanup_thread = threading.Thread(target=self.cleanup_temp_files_periodically)
        cleanup_thread.daemon = True
        cleanup_thread.start()

    def start_persistence_thread(self):
        persistence_thread = threading.Thread(target=self.persist_data)
        persistence_thread.daemon = True
        persistence_thread.start()

    def persist_data(self):
        while True:
            if self.db_lock.acquire(timeout=10):
                try:
                    # Code to persist data to the database or file
                    pass
                finally:
                    self.db_lock.release()
            else:
                logger.warning("Failed to acquire lock in persist_data")
            time.sleep(self.cleanup_interval)

    def cleanup_temp_files_periodically(self):
        while True:
            self.cleanup_tmp_files()
            time.sleep(self.cleanup_interval)

    def cleanup_tmp_files(self):
        logger.info("Starting cleanup process...")
        if self.audio_lock.acquire(timeout=10):
            try:
                tmp_files = [f for f in os.listdir('/tmp') if f.endswith('_copy.wav')]
                if not tmp_files:
                    logger.info("No files to clean up.")
                    return
                for file in tmp_files:
                    file_path = os.path.join('/tmp', file)
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted temporary file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {e}")
            finally:
                self.audio_lock.release()
        else:
            logger.warning("Failed to acquire lock in cleanup_tmp_files")
        logger.info("Cleanup process completed.")

if __name__ == "__main__":
    manager = StreamManager()
    try:
        manager.add_stream("NPR", "https://npr-ice.streamguys1.com/live.mp3")
        while True:
            time.sleep(1)
    except Exception as e:
        logger.error(f"Unhandled exception in main loop: {e}", exc_info=True)
