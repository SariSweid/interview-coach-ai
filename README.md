# Interview Coach — AI Service

Python FastAPI backend for [Interview Coach](https://github.com/SariSweid/interview-coach).
Handles speech-to-text transcription and AI answer analysis.

## Endpoints

- `GET /health` — service status check
- `POST /transcribe` — transcribes audio using Groq Whisper large-v3
- `POST /analyze` — analyzes interview answers using Anthropic Claude

## Tech Stack
- FastAPI
- Groq Whisper large-v3
- Anthropic Claude Haiku
- Python 3.14

## Environment Variables

```
ANTHROPIC_API_KEY
GROQ_API_KEY
```