from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import tempfile
import os
import json
import anthropic
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

app = FastAPI(title="Interview Coach AI Service")

# Allow requests from our Next.js frontend running on port 3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize clients — crash early if keys are missing
# Better to fail at startup than silently fail on first request
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

if not ANTHROPIC_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
if not GROQ_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in .env")

claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
groq_client = Groq(api_key=GROQ_KEY)

print("Interview Coach AI service ready!")

@app.get("/health")
def health_check():
    # Simple endpoint to confirm the service is running
    return {"status": "ok", "service": "interview-coach-ai"}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # Reject files that are too large — 25MB limit
    MAX_SIZE = 25 * 1024 * 1024
    contents = await audio.read()

    if len(contents) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="Audio file too large. Max 25MB.")

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    try:
        # Send to Groq's Whisper
        transcription = groq_client.audio.transcriptions.create(
            file=("recording.webm", contents, "audio/webm"),
            model="whisper-large-v3",
            response_format="verbose_json",
            prompt="Include all filler words and sounds like um, uh, eh, mm, hmm exactly as spoken."
        )

        return {
            "transcript": transcription.text,
            "language": transcription.language or "en"
        }

    except Exception as e:
        # Log the real error server-side, return clean message to client
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed. Please try again.")


@app.post("/analyze")
async def analyze(data: dict):
    question = data.get("question", "").strip()
    transcript = data.get("transcript", "").strip()
    language = data.get("language", "en")
    role = data.get("role", "Software Developer")
    level = data.get("level", "mid")

    # Validate required fields — don't send empty content to Claude
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is required.")

    prompt = f"""You are an expert interviewer evaluating a {level} {role} candidate.
Analyze the following interview answer and return ONLY a valid JSON object, no explanation, no markdown.

IMPORTANT: Respond in this language code: {language}
If the language is Hebrew (he) respond in Hebrew. If Arabic (ar) respond in Arabic. Otherwise respond in English.

Question asked: {question}
Candidate's answer: {transcript}

Evaluate specifically for a {level} {role} position. Consider what's expected at this level.

Return exactly this JSON shape:
{{
  "overall_score": <integer 1-10>,
  "verdict": "<one of: strong | good | needs_work | poor>",
  "dimensions": {{
    "relevance": {{ "score": <1-10>, "feedback": "<1 sentence>" }},
    "star_method": {{ "score": <1-10>, "feedback": "<1 sentence>" }},
    "specificity": {{ "score": <1-10>, "feedback": "<1 sentence>" }},
    "conciseness": {{ "score": <1-10>, "feedback": "<1 sentence>" }}
  }},
  "top_strength": "<single best thing about this answer>",
  "top_improvement": "<single most important thing to fix>"
}}"""

    try:
        message = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        # Strip markdown fences if Claude wraps the JSON in them
        text = message.content[0].text.strip()
        text = text.removeprefix("```json").removesuffix("```").strip()
        return json.loads(text)

    except json.JSONDecodeError:
        # Claude returned something that's not valid JSON
        print(f"JSON parse error. Claude returned: {text}")
        raise HTTPException(status_code=500, detail="Analysis failed — invalid response from AI.")

    except Exception as e:
        print(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")