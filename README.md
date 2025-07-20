# AutoEdit - Automatic Video Censoring Tool

An intelligent video censoring tool that automatically detects and censors inappropriate content using AI-powered transcription and analysis.

## Features

- **Automatic Transcription**: Uses OpenAI Whisper for accurate speech-to-text with word-level timestamps
- **AI-Powered Censoring**: Leverages local LLMs (Ollama) to intelligently identify content to censor
- **Smart Word Restoration**: Automatically uncensors partially censored words (e.g., "f**k" → "fuck")
- **Audio Replacement**: Replaces censored words with beep sounds while preserving video quality
- **Batch Processing**: Process multiple videos with custom censorship rules

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- [Ollama](https://ollama.ai/) installed and running locally

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd autoedit
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data:**
   ```python
   python -c "import nltk; nltk.download('words')"
   ```

4. **Set up Ollama:**
   ```bash
   # Install and start Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama serve
   
   # Pull a model (in another terminal)
   ollama pull llama3.1
   ```

5. **Configure environment (optional):**
   ```bash
   export OLLAMA_URL="http://localhost:11434"  # Default
   ```

### Usage

1. **Run the main script:**
   ```bash
   python autoedit2.py
   ```

2. **Follow the prompts:**
   - Enter the path to your video file
   - Specify what content you want to censor (e.g., "curse words", "inappropriate language")

3. **Get your results:**
   - `censored_audio.mp3` - Audio with censored words
   - `censored_video.mp4` - Final video with censored audio

## How It Works

1. **Audio Extraction**: Extracts audio from the video file
2. **Transcription**: Uses Whisper to transcribe with word-level timestamps
3. **Word Restoration**: Attempts to restore partially censored words
4. **AI Analysis**: Sends transcript to LLM to identify words to censor
5. **Audio Processing**: Replaces censored words with beep sounds
6. **Video Assembly**: Merges censored audio back with original video

## File Structure

```
autoedit/
├── autoedit2.py          # Main censoring script
├── autocensor1.py        # Alternative implementation
├── transcribe.py         # Standalone transcription utility
├── beep.wav             # Beep sound file (not included)
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Configuration

### Environment Variables

- `OLLAMA_URL`: URL for Ollama API (default: `http://localhost:11434`)

### Customization

- **Beep Sound**: Replace `beep.wav` with your preferred censoring sound
- **Whisper Model**: Change `model_size` parameter in `transcribe_with_timestamps()` for different accuracy/speed trade-offs
- **Censoring Rules**: Modify the LLM prompt in `get_censor_list_from_llm()` for custom censorship logic

## Example Usage

```bash
# Censor curse words
python autoedit2.py
# Video: my_video.mp4
# Censor: curse words and inappropriate language

# Censor specific terms
python autoedit2.py
# Video: presentation.mp4  
# Censor: company names and sensitive information
```

## Troubleshooting

### Common Issues

1. **"No face detected"**: Ensure video has clear audio
2. **Ollama connection error**: Check if Ollama is running on the correct URL
3. **CUDA out of memory**: Use smaller Whisper model or process shorter videos
4. **Beep file missing**: Download a beep sound file and name it `beep.wav`

### Performance Tips

- Use GPU for faster Whisper processing
- Process videos in smaller chunks for large files
- Use `small` Whisper model for faster processing (trade-off: accuracy)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Ollama](https://ollama.ai/) for local LLM inference
- [MoviePy](https://zulko.github.io/moviepy/) for video processing
- [NLTK](https://www.nltk.org/) for word corpus access 