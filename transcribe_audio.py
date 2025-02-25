from typing import List, Optional, Literal, Dict, Union, Sequence
from pathlib import Path
import argparse
import os
import shutil
import logging
import datetime
import sys
import subprocess
import time
import json
from enum import Enum
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import assemblyai as aai
from tqdm import tqdm  # For progress bars

load_dotenv(override=True)


class TranscriptionService(str, Enum):
    OPENAI = "openai"
    ASSEMBLYAI = "assemblyai"


# Configuration
@dataclass
class Config:
    TRANSCRIPTION_SERVICE: TranscriptionService = TranscriptionService.OPENAI
    
    # OpenAI settings
    WHISPER_MODEL: str = "whisper-1"
    
    # AssemblyAI settings
    ASSEMBLYAI_SPEAKERS_EXPECTED: Optional[int] = None  # Number of speakers (if known)
    ASSEMBLYAI_CUSTOM_SPELLING: Optional[Dict[str, Union[str, List[str]]]] = None  # Custom word mappings
    
    # General settings
    MAX_FILE_SIZE_MB: int = 25  # OpenAI's limit
    SUPPORTED_EXTENSIONS: tuple = (".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav")
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 1


class AudioTranscriber:
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        
        # Log which service we're using
        logging.info(f"Using {self.config.TRANSCRIPTION_SERVICE} for transcription")

    def setup_logging(self) -> None:
        """Configure logging with the level from environment."""
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def validate_audio_file(self, file_path: Path) -> bool:
        """Validate audio file size and format."""
        if not file_path.suffix.lower() in self.config.SUPPORTED_EXTENSIONS:
            logging.error(f"Unsupported file format: {file_path}")
            return False

        size_mb = file_path.stat().st_size / (1024 * 1024)
        # For AssemblyAI, we don't need to check file size as they support larger files
        if self.config.TRANSCRIPTION_SERVICE == TranscriptionService.OPENAI and size_mb > self.config.MAX_FILE_SIZE_MB:
            logging.error(f"File too large for OpenAI: {file_path} ({size_mb:.1f}MB)")
            return False

        return True

    def transcribe_with_openai(self, audio_file_path: Path, transcription_so_far: str) -> str:
        """Transcribe audio using OpenAI's Whisper model."""
        # Initialize OpenAI client
        client = OpenAI()
        
        with open(audio_file_path, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model=self.config.WHISPER_MODEL,
                file=audio_file,
                prompt=transcription_so_far
            )
            return response.text

    def transcribe_with_assemblyai(self, audio_file_path: Path) -> str:
        """Transcribe audio using AssemblyAI with speaker diarization."""
        # Initialize AssemblyAI client
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise ValueError("ASSEMBLYAI_API_KEY must be set in .env file")
        
        aai.settings.api_key = api_key
        
        # Build configuration parameters for AssemblyAI
        config_params = {
            "speaker_labels": True  # Enable speaker diarization
        }
        
        # Add optional parameters if they are set
        if self.config.ASSEMBLYAI_SPEAKERS_EXPECTED is not None:
            config_params["speakers_expected"] = self.config.ASSEMBLYAI_SPEAKERS_EXPECTED
            logging.info(f"Setting expected speakers to: {self.config.ASSEMBLYAI_SPEAKERS_EXPECTED}")
            
        if self.config.ASSEMBLYAI_CUSTOM_SPELLING is not None:
            config_params["custom_spelling"] = self.config.ASSEMBLYAI_CUSTOM_SPELLING
            logging.info(f"Using custom spelling with {len(self.config.ASSEMBLYAI_CUSTOM_SPELLING)} entries")
        
        # Create the configuration object
        config = aai.TranscriptionConfig(**config_params)
        
        logging.info("Using AssemblyAI with speaker diarization enabled")
        
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(str(audio_file_path), config=config)
        
        if transcript.error:
            raise RuntimeError(f"AssemblyAI transcription error: {transcript.error}")
            
        # Format transcript with speaker labels if available
        if hasattr(transcript, 'utterances') and transcript.utterances:
            formatted_text = ""
            for utterance in transcript.utterances:
                speaker = f"Speaker {utterance.speaker}"
                formatted_text += f"{speaker}: {utterance.text}\n"
            return formatted_text.strip()
        
        # Fall back to regular text if no speaker information
        return transcript.text
        
    def transcribe_audio(self, audio_file_path: Path, prompt: str) -> Optional[str]:
        """Transcribe audio file with retry mechanism."""
        logging.info(f"Transcribing audio file: {audio_file_path}")

        for attempt in range(self.config.MAX_RETRIES):
            try:
                if self.config.TRANSCRIPTION_SERVICE == TranscriptionService.OPENAI:
                    text = self.transcribe_with_openai(audio_file_path, prompt)
                else:  # AssemblyAI
                    text = self.transcribe_with_assemblyai(audio_file_path)
                
                logging.info(f'Transcription for {audio_file_path}: {len(text)} chars')
                return text
                
            except (OpenAIError, RuntimeError) as e:
                if attempt == self.config.MAX_RETRIES - 1:
                    logging.error(f"Transcription failed after {self.config.MAX_RETRIES} attempts: {e}")
                    raise
                logging.warning(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(self.config.RETRY_DELAY_SECONDS)

        return None

    def process_files(self, source_dir: Path, output_file: Path, archive_dir: Path,
                      prompt: Optional[str] = None) -> bool:
        """Process all audio files in the directory."""
        files = [f for f in source_dir.iterdir()
                 if f.suffix.lower() in self.config.SUPPORTED_EXTENSIONS]

        if not files:
            logging.error(f"No audio files found in {source_dir}")
            return False

        # Sort files by creation time
        files.sort(key=lambda x: x.stat().st_birthtime)

        transcription = ''
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for file_path in files:
                if not self.validate_audio_file(file_path):
                    continue

                try:
                    text = self.transcribe_audio(file_path, prompt or transcription)
                    if text:
                        transcription += text
                        # Move file to archive
                        shutil.move(str(file_path), str(archive_dir / file_path.name))
                        pbar.update(1)
                except Exception as e:
                    logging.error(f"Failed to process {file_path}: {e}")
                    return False

        # Save transcription
        try:
            output_file.write_text(transcription)
            final_output = archive_dir / output_file.name
            shutil.move(str(output_file), str(final_output))
            logging.info(f"Transcription saved to '{final_output}'")

            # Open Finder on macOS
            if sys.platform == "darwin":
                try:
                    subprocess.run(['open', str(archive_dir)])
                    logging.info(f"Opened Finder at '{archive_dir}'")
                except subprocess.SubprocessError as e:
                    logging.error(f"Failed to open Finder: {e}")

            return True
        except Exception as e:
            logging.error(f"Failed to save transcription: {e}")
            return False


def parse_custom_spelling(spelling_str: str) -> Dict[str, Union[str, List[str]]]:
    """Parse custom spelling string in format 'key:value1,value2,value3|key2:value1,value2'."""
    if not spelling_str:
        return {}
    
    result = {}
    
    # Split by pipe character to get key-value pairs
    pairs = spelling_str.split('|')
    
    for pair in pairs:
        if ':' not in pair:
            logging.warning(f"Ignoring invalid custom spelling pair: {pair}")
            continue
            
        key, values_str = pair.split(':', 1)
        
        # Split values by comma
        values = [v.strip() for v in values_str.split(',')]
        
        # If only one value, store as string; otherwise, store as list
        if len(values) == 1:
            result[key.strip()] = values[0]
        else:
            result[key.strip()] = values
            
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe audio files, saving transcriptions to a file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--audio-files-path',
        type=Path,
        required=True,
        help='Directory containing the audio files to process'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        help='Output file path (default: raw_transcriptions.txt in audio directory)'
    )
    parser.add_argument(
        '--prompt-file',
        type=Path,
        help='File containing the transcription prompt'
    )
    parser.add_argument(
        '--prompt',
        help='Direct transcription prompt (overrides prompt-file)'
    )
    parser.add_argument(
        '--archive-dir',
        type=Path,
        help='Directory for processed files (default: archive-TIMESTAMP in audio directory)'
    )
    parser.add_argument(
        '--service',
        type=str,
        choices=['openai', 'assemblyai'],
        default='openai',
        help='Transcription service to use (openai or assemblyai)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model to use for transcription (whisper-1 for OpenAI, nova/aurora for AssemblyAI)'
    )
    
    # AssemblyAI-specific parameters
    parser.add_argument(
        '--speakers-expected',
        type=int,
        help='Number of speakers expected in the audio (AssemblyAI only)'
    )
    parser.add_argument(
        '--custom-spelling',
        type=str,
        help='Custom spelling mappings in format "key:value1,value2,value3|key2:value1,value2" (AssemblyAI only)'
    )

    args = parser.parse_args()

    # Validate source directory
    if not args.audio_files_path.exists():
        logging.error(f"Directory does not exist: {args.audio_files_path}")
        sys.exit(1)

    # Setup paths with service-specific output filename
    service_name = args.service if args.service else "openai"
    default_output_name = f"{service_name}_raw_transcriptions.txt" 
    
    output_file = args.output_file or args.audio_files_path / default_output_name
    archive_dir = args.archive_dir or args.audio_files_path / f"archive-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    
    # Log the output file path
    logging.info(f"Transcriptions will be saved to: {output_file}")

    # Create archive directory
    archive_dir.mkdir(exist_ok=True)

    # Get prompt
    prompt = None
    if args.prompt_file and args.prompt_file.exists():
        prompt = args.prompt_file.read_text()
    if args.prompt:
        prompt = args.prompt
        logging.info("Using command-line prompt")

    if prompt:
        logging.info(f"Using prompt:\n{prompt}")

    # Configure transcription service and model
    config = Config()
    
    # Set transcription service
    if args.service:
        config.TRANSCRIPTION_SERVICE = TranscriptionService(args.service)
        
    # Set model if specified - only applicable for OpenAI
    if args.model and config.TRANSCRIPTION_SERVICE == TranscriptionService.OPENAI:
        config.WHISPER_MODEL = args.model
        
    # Note: For AssemblyAI, model selection is handled automatically on their backend
    
    # Set AssemblyAI-specific parameters
    if args.speakers_expected and config.TRANSCRIPTION_SERVICE == TranscriptionService.ASSEMBLYAI:
        config.ASSEMBLYAI_SPEAKERS_EXPECTED = args.speakers_expected
        logging.info(f"Set expected number of speakers to {args.speakers_expected}")
        
    if args.custom_spelling and config.TRANSCRIPTION_SERVICE == TranscriptionService.ASSEMBLYAI:
        try:
            custom_spelling = parse_custom_spelling(args.custom_spelling)
            config.ASSEMBLYAI_CUSTOM_SPELLING = custom_spelling
            logging.info(f"Custom spelling configured with {len(custom_spelling)} entries")
        except Exception as e:
            logging.error(f"Failed to parse custom spelling: {e}")
            sys.exit(1)
    
    # Process files
    transcriber = AudioTranscriber(config)
    success = transcriber.process_files(args.audio_files_path, output_file, archive_dir, prompt)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()