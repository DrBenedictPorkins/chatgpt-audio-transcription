import argparse
import os
import shutil
import logging
import datetime
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv(override=True)

client = OpenAI()

WHISPER_MODEL = "whisper-1"

# set logging level based on environment variable LOG_LEVEL defaulting to INFO
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def test_can_write_to_output_file(output_file):
    with open(output_file, "w") as f:
        pass
    os.remove(output_file)


def transcribe_audio(audio_file_path, transcription_so_far):
    logging.info(f"Transcribing audio file: {audio_file_path}")
    try:
        with open(audio_file_path, 'rb') as audio_file:
            rs = client.audio.transcriptions.create(model=WHISPER_MODEL, file=audio_file,
                                                    prompt=transcription_so_far)

        text = rs.text
        logging.info(f'Transcription for {audio_file_path}: {len(text)}')
        return text
    except OpenAIError as e:
        logging.error(f"Transcription failed: {e}")
        raise


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Transcribe audio files, saving transcriptions to a file.')

    arg_parser.add_argument('--audio-files-path', dest='audio_files_path', type=str,
                            help='The path to the directory containing the audio files to process.', required=True)
    arg_parser.add_argument('--output-file', dest='output_file', type=str, required=False)
    arg_parser.add_argument('--prompt-file', dest='prompt_file', type=str, required=False)
    arg_parser.add_argument('--prompt', dest='prompt', type=str, required=False,
                            help='The prompt to use for transcription, overrides --prompt-file if both are specified.')
    arg_parser.add_argument('--archive-dir', dest='archive_dir', type=str, required=False)

    args = arg_parser.parse_args()

    source_dir = args.audio_files_path
    if not os.path.exists(source_dir):
        logging.error("Directory does not exist: " + source_dir)
        exit()

    output_file = args.output_file or os.path.join(source_dir, "raw_transcriptions.txt")

    if os.path.exists(output_file):
        logging.error("Output file already exists: " + output_file)
        exit()

    test_can_write_to_output_file(output_file)

    extensions = [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a"]
    files = [f for f in os.listdir(source_dir) if any(f.endswith(ext) for ext in extensions)]

    files.sort(key=lambda x: os.path.getctime(os.path.join(source_dir, x)), reverse=False)

    if len(files) == 0:
        logging.error("No audio files found in " + source_dir)
        exit()

    prompt = None
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()

    if args.prompt:
        logging.info("Using command-line args prompt.")
        prompt = args.prompt

    if prompt is None:
        logging.info("Prompt not specified.")
    else:
        logging.info("Using prompt: \n\n" + prompt)

    archive_dir = args.archive_dir or os.path.join(source_dir,
                                                   f"archive-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    os.mkdir(archive_dir)
    logging.info(f"Created archive directory: {archive_dir}")

    transcription = ''
    for filename in files:
        full_path = os.path.join(source_dir, filename)

        transcription += transcribe_audio(full_path, prompt or transcription)
        logging.info(f"Transcription length so far: {len(transcription)}")

        shutil.move(full_path, os.path.join(archive_dir, filename))

    logging.info(f"All audio files in '{source_dir}' have been processed and moved to '{archive_dir}'.")
    with open(output_file, 'w') as f:
        f.write(transcription)

    logging.info(f"Transcription saved to '{output_file}'.")
    logging.info("Done, have a great weekend!")
