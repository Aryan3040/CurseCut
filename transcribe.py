import whisper
from moviepy import VideoFileClip
import torch
import nltk
from difflib import get_close_matches
import re

nltk.download('words')
english_vocab = set(nltk.corpus.words.words())

def extract_audio(video_path, audio_output):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_output)
    video.close()

def transcribe_with_timestamps(audio_path, model_size='small'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = whisper.load_model(model_size).to(device)
    result = model.transcribe(audio_path, word_timestamps=True)

    words_with_timestamps = []
    for segment in result['segments']:
        for word in segment['words']:
            words_with_timestamps.append({
                'start': word['start'],
                'end': word['end'],
                'word': word['word']
            })
    return words_with_timestamps

def uncensor_words(words_list):
    uncensored_transcript = []
    for item in words_list:
        word = item['word']
        # Check if word contains asterisks
        if '*' in word:
            # Create regex pattern from censored word
            pattern = '^' + word.replace('*', '.') + '$'
            matches = [w for w in english_vocab if re.match(pattern, w, re.IGNORECASE)]
            
            if matches:
                # Choose the most common similar word
                replacement = matches[0]
            else:
                # If no exact length match, use close match
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

def print_transcript(transcript):
    for word_info in transcript:
        print(f"{word_info['start']:.2f} - {word_info['end']:.2f}: {word_info['word']}")

if __name__ == "__main__":
    video_file = "testMovie.mp4"
    audio_file = "extracted_audio.mp3"

    print("Extracting audio from video...")
    extract_audio(video_file, audio_file)

    print("\nTranscribing audio with Whisper (small model)...")
    words_list = transcribe_with_timestamps(audio_file, model_size='small')

    print("\nRestoring censored words...")
    full_transcript = uncensor_words(words_list)

    print("\nFinal Uncensored Transcript with Timestamps:\n")
    print_transcript(full_transcript)
