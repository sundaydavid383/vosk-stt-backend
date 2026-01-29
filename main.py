from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from vosk import Model, KaldiRecognizer
import os, threading, urllib.request, zipfile, json
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool

# -------------------------------
# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODELS_DIR = os.path.join(BASE_DIR, "models")
VOSK_MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
SAMPLE_RATE = 16000
MAX_BODY_SIZE = 5 * 1024 * 1024  # 5 MB limit

print("🟢 Backend starting...", flush=True)
print("📁 BASE_DIR:", BASE_DIR, flush=True)
print("📁 MODEL PATH:", VOSK_MODEL_PATH, flush=True)

# -------------------------------
# Thread lock for safe model loading
model_lock = threading.Lock()
_model_instance = None

def download_and_extract_model():
    if os.path.exists(VOSK_MODEL_PATH):
        print("✅ Model already exists", flush=True)
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    zip_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}.zip")

    print("⬇️ Downloading Vosk model...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(MODELS_DIR)

    os.remove(zip_path)
    print("✅ Model extracted", flush=True)

def get_model():
    global _model_instance
    if _model_instance is None:
        with model_lock:
            if _model_instance is None:
                print("🔹 Loading Vosk model...", flush=True)
                download_and_extract_model()
                _model_instance = Model(VOSK_MODEL_PATH)
                print("✅ Vosk model loaded", flush=True)
    return _model_instance

# -------------------------------
# Vosk wrapper
class VoskRecognizer:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def transcribe(self, pcm_bytes: bytes) -> str:
        model = get_model()
        rec = KaldiRecognizer(model, self.sample_rate)
        rec.AcceptWaveform(pcm_bytes)
        result = json.loads(rec.FinalResult())
        return result.get("text", "")

vosk = VoskRecognizer(SAMPLE_RATE)

# -------------------------------
# Lifespan (modern FastAPI startup)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Preloading Vosk model on startup...", flush=True)
    get_model()
    print("✅ Vosk model ready", flush=True)
    yield
    print("🛑 Application shutting down...", flush=True)

app = FastAPI(lifespan=lifespan)

# -------------------------------
# Middleware to limit request size
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = int(request.headers.get("content-length", 0))
    if content_length > MAX_BODY_SIZE:
        return JSONResponse({"detail": "Request too large"}, status_code=413)
    response = await call_next(request)
    return response

# -------------------------------
# Health check
@app.get("/")
def root():
    return {"service": "vosk-stt", "status": "running"}

# -------------------------------
# Speech-to-text endpoint
@app.post("/stt")
async def stt_endpoint(
    request: Request,
    x_user_id: str = Header(...),
    x_session_id: str = Header(...),
    x_mode: str = Header("vosk"),
):
    try:
        audio_bytes = await request.body()
        print(f"🔹 Received audio ({len(audio_bytes)} bytes) from session {x_session_id}", flush=True)

        if len(audio_bytes) < 44:
            raise ValueError("Audio too short")

        # Strip WAV header (44 bytes)
        pcm_bytes = audio_bytes[44:]

        # Run transcription in a threadpool (non-blocking)
        transcript = await run_in_threadpool(vosk.transcribe, pcm_bytes)

        print(f"✅ Transcript ready for session {x_session_id}: {transcript}", flush=True)

        return {"transcript": transcript, "sessionId": x_session_id}

    except Exception as e:
        print("❌ STT ERROR:", repr(e), flush=True)
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
if __name__ == "__main__":
    import uvicorn
    port_str = os.environ.get("PORT", "8765")
    port = int(port_str) if port_str.isdigit() else 8765
    print(f"🚀 Starting server on port {port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
