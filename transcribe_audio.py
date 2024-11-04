from typing import List, Optional
from pathlib import Path
import argparse
import os
import shutil
import logging
import datetime
import sys
import subprocess
import time  # Added time module import
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from tqdm import tqdm  # For progress bars

load_dotenv(override=True)


# Configuration
@dataclass
class Config:
    WHISPER_MODEL: str = "whisper-1"
    MAX_FILE_SIZE_MB: int = 25  # OpenAI's limit
    SUPPORTED_EXTENSIONS: tuple = (".mp3", ".mp4", ".mpeg", ".mpga", ".m4a")
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 1


class AudioTranscriber:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI()
        self.setup_logging()

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
        if size_mb > self.config.MAX_FILE_SIZE_MB:
            logging.error(f"File too large: {file_path} ({size_mb:.1f}MB)")
            return False

        return True

    def transcribe_audio(self, audio_file_path: Path, transcription_so_far: str) -> Optional[str]:
        """Transcribe audio file with retry mechanism."""
        logging.info(f"Transcribing audio file: {audio_file_path}")

        for attempt in range(self.config.MAX_RETRIES):
            try:
                with open(audio_file_path, 'rb') as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model=self.config.WHISPER_MODEL,
                        file=audio_file,
                        prompt=transcription_so_far
                    )
                    text = response.text
                    logging.info(f'Transcription for {audio_file_path}: {len(text)} chars')
                    return text
            except OpenAIError as e:
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

    args = parser.parse_args()

    # Validate source directory
    if not args.audio_files_path.exists():
        logging.error(f"Directory does not exist: {args.audio_files_path}")
        sys.exit(1)

    # Setup paths
    output_file = args.output_file or args.audio_files_path / "raw_transcriptions.txt"
    archive_dir = args.archive_dir or args.audio_files_path / f"archive-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

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

    # Process files
    transcriber = AudioTranscriber(Config())
    success = transcriber.process_files(args.audio_files_path, output_file, archive_dir, prompt)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()