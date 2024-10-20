# Audio Transcription Project

This project is a Python script that transcribes audio files using OpenAI's Whisper model. It processes all audio files in a specified directory, transcribes them, and saves the transcriptions to a single output file.

## Features

- Transcribes multiple audio file formats (mp3, mp4, mpeg, mpga, m4a)
- Supports custom prompts for improved transcription accuracy
- Archives processed audio files
- Configurable logging

## Prerequisites

- Python 3.7 or higher
- OpenAI API key

## Installation

1. Clone this repository or download the source code.
2. Navigate to the project directory.
3. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the script with the following command:

```
python transcribe_audio.py --audio-files-path /path/to/audio/files
```

### Optional arguments:

- `--output-file`: Specify the output file for transcriptions (default: raw_transcriptions.txt in the audio files directory)
- `--prompt-file`: Path to a file containing a prompt for transcription
- `--prompt`: Directly specify a prompt for transcription (overrides --prompt-file if both are specified)
- `--archive-dir`: Specify a custom archive directory for processed audio files

Example with all options:

```
python transcribe_audio.py --audio-files-path /path/to/audio/files --output-file /path/to/output.txt --prompt "This conversation may include names like Makram, John Doe, Jane Smith, and technologies or terms such as Python, JavaScript, OpenAI, ChatGPT, Whisper, machine learning, and audio transcription." --archive-dir /path/to/archive
```

## Logging

The script uses Python's logging module. You can set the log level by setting the `LOG_LEVEL` environment variable (default: INFO).

## License

This project is open-source and available under the MIT License.
