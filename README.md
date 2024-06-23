
# Radio Analyzer

Radio Analyzer is a Django-based application that streams audio from online radio stations, processes the audio for transcription using speech recognition, and generates summaries of the transcribed text using a natural language processing model. The application stores transcriptions and summaries in a database and provides an interface for viewing and managing these data.

## Features

- **Stream Audio from Online Radio Stations**: Uses `ffmpeg` to stream audio in real-time from specified URLs.
- **Speech Recognition**: Utilizes the `speech_recognition` library to transcribe audio into text.
- **Audio Normalization**: Normalizes audio using `pydub` to ensure consistent volume levels.
- **Summarization**: Uses a pre-trained BART model from Hugging Face's Transformers library to summarize the transcribed text.
- **Data Storage**: Stores transcriptions and summaries in a PostgreSQL database.
- **Web Interface**: Provides a Django web interface for managing and viewing transcriptions and summaries.
- **Error Handling and Logging**: Implements robust error handling and logging throughout the process.

## Installation

### Prerequisites

- Python 3.11
- PostgreSQL
- FFmpeg

### Setting Up the Environment

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/radio_analyzer.git
    cd radio_analyzer
    ```

2. **Set up a Python virtual environment**:
    ```bash
    python -m venv env
    source env/bin/activate
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Install FFmpeg**:
    - On Ubuntu:
        ```bash
        sudo apt update
        sudo apt install ffmpeg
        ```
    - On MacOS (using Homebrew):
        ```bash
        brew install ffmpeg
        ```

### Database Configuration

1. **Create a PostgreSQL database**:
    ```bash
    sudo -u postgres psql
    CREATE DATABASE radio_analyzer;
    CREATE USER radio_user WITH PASSWORD 'password';
    ALTER ROLE radio_user SET client_encoding TO 'utf8';
    ALTER ROLE radio_user SET default_transaction_isolation TO 'read committed';
    ALTER ROLE radio_user SET timezone TO 'UTC';
    GRANT ALL PRIVILEGES ON DATABASE radio_analyzer TO radio_user;
    \q
    ```

2. **Apply the migrations**:
    ```bash
    python manage.py migrate
    ```

### Configuration

1. **Environment Variables**:
    Create a `.env` file in the project root directory and add the following configurations:
    ```env
    DATABASE_NAME=radio_analyzer
    DATABASE_USER=radio_user
    DATABASE_PASSWORD=password
    DATABASE_HOST=localhost
    DATABASE_PORT=5432
    ```

2. **Update Django settings**:
    Modify `radio_analyzer/settings.py` to use environment variables for database settings.

## Usage

### Running the Development Server

1. **Start the Django development server**:
    ```bash
    python manage.py runserver
    ```

2. **Access the application**:
    Open a web browser and go to `http://127.0.0.1:8000/`.

### Managing Streams

1. **Adding a Stream**:
    Use the Django admin interface to add a new radio station stream. Navigate to `http://127.0.0.1:8000/admin/` and log in with your admin credentials. Add a new `RadioStation` with the desired name and URL.

2. **Starting the Stream**:
    The stream manager will automatically start processing the stream once it is added.

## Project Structure

- **radio_analyzer/**: Main Django project directory.
    - **settings.py**: Django settings configuration.
    - **urls.py**: URL routing configuration.
- **streams/**: Django app for handling streams.
    - **models.py**: Database models for radio stations, transcriptions, and segments.
    - **views.py**: Views for the web interface.
    - **utils/stream_manager.py**: Core logic for streaming, transcribing, and summarizing audio.

## Logging

- Logs are configured to output to the console by default. You can configure additional log handlers in `radio_analyzer/settings.py`.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Django](https://www.djangoproject.com/)
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)
- [Pydub](https://pydub.com/)
- [Transformers](https://huggingface.co/transformers/)
- [FFmpeg](https://ffmpeg.org/)
