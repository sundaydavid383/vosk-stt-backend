from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from vosk import Model, KaldiRecognizer
import os
import threading
import json
import gc
import asyncio
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool
from io import BytesIO
import wave

# -------------------------------
# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODELS_DIR = os.path.join(BASE_DIR, "model")
VOSK_MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
SAMPLE_RATE = 16000
MAX_BODY_SIZE = 5 * 1024 * 1024  # 5 MB max per request

# -------------------------------
# Thread-safe model loading
_model_lock = threading.Lock()
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                if not os.path.exists(VOSK_MODEL_PATH):
                    raise RuntimeError("Vosk model not found! Make sure it is downloaded and unzipped in /models")
                print("🔹 Loading Vosk model...")
                _model_instance = Model(VOSK_MODEL_PATH)
                print("✅ Vosk model loaded")
    return _model_instance

# -------------------------------
# Limit concurrent transcriptions per worker
# Adjust this number based on your CPU cores
transcription_semaphore = asyncio.Semaphore(1)

# -------------------------------
class VoskRecognizer:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def validate_wav(self, wav_bytes: bytes):
        """
        Validate that WAV is mono, 16-bit PCM, correct sample rate.
        Throws ValueError if invalid.
        """
        try:
            with wave.open(BytesIO(wav_bytes), "rb") as wf:
                if wf.getnchannels() != 1:
                    raise ValueError("Only mono audio supported")
                if wf.getsampwidth() != 2:
                    raise ValueError("Only 16-bit audio supported")
                if wf.getframerate() != self.sample_rate:
                    raise ValueError(f"Sample rate must be {self.sample_rate}Hz")
        except wave.Error:
            raise ValueError("Invalid WAV data")

    def transcribe(self, pcm_bytes: bytes) -> str:
        """
        Transcribe audio in small chunks to avoid memory spikes.
        """
        model = get_model()
        rec = KaldiRecognizer(model, self.sample_rate)
        chunk_size = 32 * 1024
        for i in range(0, len(pcm_bytes), chunk_size):
            rec.AcceptWaveform(pcm_bytes[i:i+chunk_size])
        result = json.loads(rec.FinalResult())
        return result.get("text", "")

vosk = VoskRecognizer(SAMPLE_RATE)

# -------------------------------
# FastAPI lifespan to preload model
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Preloading Vosk model on startup...")
    get_model()
    print("✅ Vosk model ready")
    yield
    print("🛑 Application shutting down...")

app = FastAPI(lifespan=lifespan)

# -------------------------------
# Middleware to limit request size
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = int(request.headers.get("content-length", 0))
    if content_length > MAX_BODY_SIZE:
        return JSONResponse({"detail": "Request too large"}, status_code=413)
    return await call_next(request)

# -------------------------------
# Health check
@app.get("/")
def root():
    return {"service": "vosk-stt", "status": "running"}

# -------------------------------
# STT endpoint
@app.post("/stt")
async def stt_endpoint(
    request: Request,
    x_user_id: str = Header(...),
    x_session_id: str = Header(...),
    x_mode: str = Header("vosk"),
):
    try:
        audio_bytes = await request.body()
        print(f"🔹 Received audio ({len(audio_bytes)} bytes) from session {x_session_id}")

        if len(audio_bytes) < 44:
            raise HTTPException(status_code=400, detail="Audio too short")

        # Optional WAV validation
        try:
            vosk.validate_wav(audio_bytes)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        # Strip WAV header for PCM
        pcm_bytes = bytes(audio_bytes[44:])

        # Limit concurrent transcription per worker
        async with transcription_semaphore:
            transcript = await run_in_threadpool(vosk.transcribe, pcm_bytes)

        # Clean memory
        del audio_bytes, pcm_bytes
        gc.collect()

        print(f"✅ Transcript ready for session {x_session_id}: {transcript}")
        return {"transcript": transcript, "sessionId": x_session_id}

    except HTTPException as he:
        raise he
    except Exception as e:
        print("❌ STT ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# ⚠️ NOTE: DO NOT call uvicorn.run() in production
# On Render or other cloud platforms, start with:
# uvicorn app:app --host 0.0.0.0 --port $PORT --workers 4
# Adjust number of workers based on CPU cores and expected load

