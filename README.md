# Audio Transcription Project

This project is a Python script that transcribes audio files using either OpenAI's Whisper model or AssemblyAI's transcription service. It processes all audio files in a specified directory, transcribes them, and saves the transcriptions to a single output file.

## Features

- Supports multiple transcription services (OpenAI and AssemblyAI)
- Transcribes multiple audio file formats (mp3, mp4, mpeg, mpga, m4a, wav)
- Supports custom prompts for improved transcription accuracy (OpenAI)
- Archives processed audio files
- Configurable logging

## Prerequisites

- Python 3.7 or higher
- OpenAI API key (if using OpenAI service)
- AssemblyAI API key (if using AssemblyAI service)

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
5. Create a `.env` file in the project root and add your API key(s):
   ```
   # For OpenAI
   OPENAI_API_KEY=your_openai_api_key_here
   
   # For AssemblyAI
   ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
   ```

## Usage

Run the script with the following command:

```
python transcribe_audio.py --audio-files-path /path/to/audio/files
```

### Optional arguments:

- `--output-file`: Specify the output file for transcriptions (default: raw_transcriptions.txt in the audio files directory)
- `--prompt-file`: Path to a file containing a prompt for transcription (for OpenAI)
- `--prompt`: Directly specify a prompt for transcription (overrides --prompt-file if both are specified)
- `--archive-dir`: Specify a custom archive directory for processed audio files
- `--service`: Transcription service to use (`openai` or `assemblyai`, default: openai)
- `--model`: Model to use for transcription (default: whisper-1 for OpenAI, nova for AssemblyAI)

Example with OpenAI:

```
python transcribe_audio.py --audio-files-path /path/to/audio/files --service openai --output-file /path/to/output.txt --prompt "This conversation may include names like Makram, John Doe, Jane Smith, and technologies or terms such as Python, JavaScript, OpenAI, ChatGPT, Whisper, machine learning, and audio transcription." --archive-dir /path/to/archive
```

Example with AssemblyAI:

```
python transcribe_audio.py --audio-files-path /path/to/audio/files --service assemblyai --model aurora --output-file /path/to/output.txt --archive-dir /path/to/archive
```

## Logging

The script uses Python's logging module. You can set the log level by setting the `LOG_LEVEL` environment variable (default: INFO).

## License

This project is open-source and available under the MIT License.
