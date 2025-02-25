# Audio Transcription Project Guidelines

## Commands
- **Setup**: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- **Run**: `python transcribe_audio.py --audio-files-path /path/to/audio/files`
- **Test**: Use Python's built-in testing modules: `python -m unittest discover`

## Code Style
- **Formatting**: Use PEP 8 with 4-space indentation, 100 char line length
- **Imports**: Group standard library, third-party, and local imports with one blank line between groups
- **Types**: Use type hints for all functions and methods; use `Optional` for nullable types
- **Naming**: 
  - Classes: PascalCase
  - Functions/variables: snake_case
  - Constants: UPPER_SNAKE_CASE
- **Error handling**: Use specific exception types with meaningful error messages; implement retry mechanisms
- **Logging**: Use Python's logging module with appropriate levels (INFO, ERROR, etc.)
- **Documentation**: Docstrings for all classes and functions using triple quotes

## Architecture
The codebase is centered around the `AudioTranscriber` class, which handles the process of transcribing audio files using OpenAI's Whisper model. Configuration is managed via dataclasses.