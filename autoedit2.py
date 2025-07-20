import os
import re
import json
import nltk
import torch
import whisper
import requests

from difflib import get_close_matches
from moviepy import VideoFileClip, AudioFileClip
from pydub import AudioSegment

# Ensure you have downloaded the "words" corpus for NLTK
nltk.download('words')
english_vocab = set(nltk.corpus.words.words())

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")  # Configure via OLLAMA_URL env var
BEEP_FILE = "beep.wav"                 # Place your beep file here

def extract_audio(video_path, audio_output):
    """Extract audio from the given video using MoviePy."""
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_output, fps=16000)  # 16k sample rate for Whisper
    video.close()

def transcribe_with_timestamps(audio_path, model_size='small'):
    """
    Transcribe audio with OpenAI Whisper, retrieving word-level timestamps.
    Returns a list of dicts: [{'start': float, 'end': float, 'word': str}, ...]
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Whisper on device: {device}")
    model = whisper.load_model(model_size).to(device)

    # 'word_timestamps=True' requires the 'verbose_json' decoding
    result = model.transcribe(audio_path, word_timestamps=True)
    words_with_timestamps = []
    for segment in result['segments']:
        for w in segment['words']:
            words_with_timestamps.append({
                'start': w['start'],
                'end': w['end'],
                'word': w['word']
            })
    return words_with_timestamps

def uncensor_words(words_list):
    """
    If words are partially censored (e.g., 'f**k'), attempt to guess the full word
    from the English corpus. If no direct match is found, try close matches.
    """
    uncensored_transcript = []
    for item in words_list:
        word = item['word']
        if '*' in word:
            # Create regex pattern from censored word (replace '*' with '.')
            pattern = '^' + word.replace('*', '.') + '$'
            matches = [w for w in english_vocab if re.match(pattern, w, re.IGNORECASE)]

            if matches:
                replacement = matches[0]
            else:
                # Try close matches by removing asterisks
                clean_word = word.replace('*', '')
                close_matches = get_close_matches(clean_word, english_vocab, n=1, cutoff=0.6)
                replacement = close_matches[0] if close_matches else word
            uncensored_transcript.append({
                'start': item['start'],
                'end': item['end'],
                'word': replacement
            })
        else:
            uncensored_transcript.append(item)
    return uncensored_transcript

def get_censor_list_from_llm(transcript_text, censor_request):
    """
    Sends the transcript and user’s censorship request to an LLM (Ollama example).
    Expects the LLM to return a JSON list of words to censor, e.g.: ["word1","word2"].
    Adjust prompts/model as needed for your environment.
    """
    # Refined system prompt to instruct the LLM to ONLY return words/phrases to censor
    system_prompt = (
        "You are a text-processing assistant. A user wants to censor specific content from a transcript. "
        "Your job is to identify and list only the words or phrases from the transcript that need to be censored. "
        "Do not provide explanations, commentary, or additional context—only return a JSON list of the words/phrases "
        "that should be censored based on the user's censorship request."
        "Example of response: [\"curse\", \"word\", \"explicit\"]"
    )

    final_prompt = f"{system_prompt}\n\nTranscript:\n{transcript_text}\n\nUser's Censorship Request:\n{censor_request}\n\nAssistant:"

    data = {
        "model": "llama3.1",  # or whatever model is available in Ollama
        "prompt": final_prompt,
        "stream": False
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=data, headers=headers)
    response.raise_for_status()
    result = response.json()

    raw_text = result.get("response", "").strip()

    # Attempt to parse JSON list
    try:
        censor_list = json.loads(raw_text)
    except:
        censor_list = []

    return censor_list

def beep_out_words(audio_path, words_list, words_to_censor, output_path):
    """
    Replaces the audio segment of each censored word with a beep using pydub.
      - words_list is [{'start': float, 'end': float, 'word': str}, ...]
      - words_to_censor is a set of strings (lowercase).
    """
    original_audio = AudioSegment.from_file(audio_path)
    beep_sound = AudioSegment.from_file(BEEP_FILE)

    # Process each word in chronological order
    for w in sorted(words_list, key=lambda x: x['start']):
        word_lower = w['word'].lower().strip()
        if word_lower in words_to_censor:
            start_ms = int(w['start'] * 1000)
            end_ms = int(w['end'] * 1000)
            duration_ms = end_ms - start_ms

            # Make a beep segment of matching length
            segment_to_use = beep_sound
            if len(beep_sound) < duration_ms:
                # Loop beep until it covers the entire word
                times = (duration_ms // len(beep_sound)) + 1
                segment_to_use = (beep_sound * times)[:duration_ms]
            else:
                # Just slice beep to match
                segment_to_use = beep_sound[:duration_ms]

            # Overlay beep onto original audio
            original_audio = original_audio.overlay(segment_to_use, position=start_ms)

    # Export final
    original_audio.export(output_path, format="mp3")

def merge_audio_with_video(video_path, audio_path, output_path):
    """Merges the given audio with the original video."""
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    # Set the audio of the video to the censored audio
    final_video = video.with_audio(audio)
    
    # Write the final video with the new audio
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

def main():
    # 1) Get user inputs
    video_path = input("Enter the path to your video file: ").strip()
    censor_request = input("Enter what you want to censor: ").strip()

    # Check file exists
    if not os.path.isfile(video_path):
        print(f"Error: video file '{video_path}' not found.")
        return

    # 2) Extract audio
    temp_audio = "temp_audio.wav"
    print("\nExtracting audio from video...")
    extract_audio(video_path, temp_audio)
    print("Audio extracted.\n")

    # 3) Transcribe with timestamps
    print("Transcribing audio with Whisper...")
    words_list = transcribe_with_timestamps(temp_audio, model_size='small')
    print("Transcription complete.\n")

    # 4) Attempt to restore censored words (f**k -> 'fuck' if guessed)
    words_list = uncensor_words(words_list)

    # 5) Convert entire transcript to text
    full_transcript_text = " ".join([w['word'] for w in words_list])
    print("Full transcript:")
    print(full_transcript_text)
    print("\n")

    # 6) Call the LLM to get list of words to censor
    print("Contacting LLM to figure out which words to censor...")
    words_to_censor = get_censor_list_from_llm(full_transcript_text, censor_request)
    print(f"LLM suggests censoring these words/phrases: {words_to_censor}\n")

    if not words_to_censor:
        print("No words to censor. Exiting.")
        return

    # 7) Beep out the words
    censored_output = "censored_audio.mp3"
    print(f"Beeping out {len(words_to_censor)} items in {temp_audio}...")
    beep_out_words(temp_audio, words_list, set(w.lower() for w in words_to_censor), censored_output)
    print(f"Censored audio saved to: {censored_output}")

    # (Optional) Merge censored audio back with video
    censored_video_output = "censored_video.mp4"
    merge_audio_with_video(video_path, censored_output, censored_video_output)
    print(f"Censored video saved to: {censored_video_output}")

if __name__ == "__main__":
    main()
