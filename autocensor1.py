import os
import re
import json
import nltk
import torch
import whisper
import requests
import shutil

from difflib import get_close_matches
from moviepy import VideoFileClip, AudioFileClip
from pydub import AudioSegment

# Ensure you have downloaded the "words" corpus for NLTK
nltk.download('words')
english_vocab = set(nltk.corpus.words.words())

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")  # Configure via OLLAMA_URL env var
BEEP_FILE = "beep.wav"                 # Place your beep file here

# A fallback dictionary of common swear words
COMMON_SWEARS = {
    "fuck", "fucks", "fucking", "shit", "shits", "bitch", "bitches", "bastard", "bastards", "asshole", "assholes",
    "whore", "whores", "dick", "dicks", "cock", "cocks", "piss", "pissed", "cunt", "slut", "sluts", "faggot", "faggots",
    "motherfucker", "motherfuckers", "goddamn", "damn", "damnit", "nigger", "nigga", "niggers", "dicking", "bullshit"
}


def extract_audio(video_path, audio_output):
    """Extract audio from the given video using MoviePy."""
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_output, fps=16000)
    video.close()


def transcribe_with_timestamps(audio_path, model_size='small'):
    """
    Transcribe audio with OpenAI Whisper, retrieving word-level timestamps.
    Returns a list of dicts: [{'start': float, 'end': float, 'word': str}, ...]
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Whisper on device: {device}")
    model = whisper.load_model(model_size).to(device)

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
    If words are partially censored (e.g., 'f**k'), attempt to guess the full word.
    """
    uncensored_transcript = []
    for item in words_list:
        word = item['word']
        if '*' in word:
            pattern = '^' + word.replace('*', '.') + '$'
            matches = [w for w in english_vocab if re.match(pattern, w, re.IGNORECASE)]

            if matches:
                replacement = matches[0]
            else:
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
    Sends the transcript and user’s censorship request to an LLM.
    Expects a JSON list of words to censor.
    """
    system_prompt = (
        "You are a text-processing assistant. A user wants to censor specific content from a transcript. "
        "Your job is to identify and list only the words or phrases from the transcript that need to be censored. "
        "Do not provide explanations, commentary, or additional context—only return a JSON list of the words/phrases "
        "that should be censored based on the user's censorship request."\
        "Example of response: [\"curse\", \"word\", \"explicit\"]"
    )

    final_prompt = f"{system_prompt}\n\nTranscript:\n{transcript_text}\n\nUser's Censorship Request:\n{censor_request}\n\nAssistant:"

    data = {
        "model": "llama3.1",  # or your preferred model
        "prompt": final_prompt,
        "stream": False
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=data, headers=headers)
    response.raise_for_status()
    result = response.json()

    raw_text = result.get("response", "").strip()
    try:
        censor_list = json.loads(raw_text)
    except:
        censor_list = []

    return censor_list


def fallback_swears(words_list):
    """
    Identify any common swear words from our dictionary that appear in the transcript.
    """
    found = set()
    for w in words_list:
        w_lower = w['word'].lower().strip(".,!?-\"'\n\r")
        if w_lower in COMMON_SWEARS:
            found.add(w_lower)
    return found


def beep_out_words(audio_path, words_list, words_to_censor, output_path):
    """
    Overlays a beep for each censored word. Removes original audio portion with silence.
    """
    original_audio = AudioSegment.from_file(audio_path)
    beep_sound = AudioSegment.from_file(BEEP_FILE)

    # Decrease beep volume by 10dB if desired.
    beep_sound = beep_sound - 0

    for w in sorted(words_list, key=lambda x: x['start']):
        word_lower = w['word'].lower().strip(".,!?-\"'\n\r")
        if word_lower in words_to_censor:
            start_ms = int(w['start'] * 1000)
            end_ms = int(w['end'] * 1000)
            duration_ms = end_ms - start_ms
            if duration_ms < 50:
                continue

            # Slice or loop beep to match the exact duration
            if len(beep_sound) < duration_ms:
                times = (duration_ms // len(beep_sound)) + 1
                beep_segment = (beep_sound * times)[:duration_ms]
            else:
                beep_segment = beep_sound[:duration_ms]

            # 1) overlay silence
            silence_segment = AudioSegment.silent(duration=duration_ms)
            original_audio = original_audio.overlay(silence_segment, position=start_ms)

            # 2) overlay beep
            original_audio = original_audio.overlay(beep_segment, position=start_ms)

    original_audio.export(output_path, format="mp3")


def merge_audio_with_video(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    final_video = video.with_audio(audio)
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")


def main():
    video_path = input("Enter the path to your video file: ").strip()
    censor_request = input("Enter what you want to censor: ").strip()

    if not os.path.isfile(video_path):
        print(f"Error: video file '{video_path}' not found.")
        return

    temp_audio = "temp_audio.wav"
    print("\nExtracting audio from video...")
    extract_audio(video_path, temp_audio)
    print("Audio extracted.\n")

    print("Transcribing audio with Whisper...")
    words_list = transcribe_with_timestamps(temp_audio, model_size='small')
    print("Transcription complete.\n")

    # Uncensor partial words
    words_list = uncensor_words(words_list)

    # Combine transcript
    full_transcript_text = " ".join([w['word'] for w in words_list])
    print("Full transcript:")
    print(full_transcript_text)
    print("\n")

    print("Contacting LLM to figure out which words to censor...")
    llm_censor_words = get_censor_list_from_llm(full_transcript_text, censor_request)

    # Only do fallback if user specifically wants to censor curse words
    combined_censor_set = set(w.lower() for w in llm_censor_words)

    # If user mentions 'curse' or 'swear' in their request, use fallback
    if re.search(r"\b(curse|swear)\b", censor_request.lower()):
        fallback_curses = fallback_swears(words_list)
        combined_censor_set = combined_censor_set | fallback_curses

    print(f"LLM suggests censoring: {llm_censor_words}")
    print(f"Final combined list: {sorted(list(combined_censor_set))}\n")

    if not combined_censor_set:
        print("No words to censor. Exiting.")
        return

    censored_output = "censored_audio.mp3"
    print(f"Beeping out {len(combined_censor_set)} items...")
    beep_out_words(temp_audio, words_list, combined_censor_set, censored_output)
    print(f"Censored audio saved to: {censored_output}")

    base, ext = os.path.splitext(video_path)
    censored_video_output = f"{base}_censored{ext}"

    merge_audio_with_video(video_path, censored_output, censored_video_output)
    print(f"Censored video saved to: {censored_video_output}")

    if not os.path.exists("data"):
        os.makedirs("data")

    try:
        shutil.move(temp_audio, os.path.join("data", temp_audio))
    except Exception as e:
        print(f"Could not move {temp_audio}: {e}")

    try:
        shutil.move(censored_output, os.path.join("data", censored_output))
    except Exception as e:
        print(f"Could not move {censored_output}: {e}")

    print("\nAll temporary files have been moved to the 'data' folder.")

if __name__ == "__main__":
    main()
