from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import VitsModel, AutoTokenizer
import torch
import io
import scipy.io.wavfile as wavfile
import os

app = FastAPI(
    title="Kinyarwanda TTS API",
    description="Text-to-speech API using facebook/mms-tts-kin for Kinyarwanda",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None

class TTSRequest(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global model, tokenizer
    model_name = os.getenv("TTS_MODEL_NAME", "facebook/mms-tts-kin")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = VitsModel.from_pretrained(model_name)
        model.to("cpu")
        model.eval()
    except Exception as e:
        print("Error loading model:", e)

@app.get("/health")
def health():
    status = "ok" if model is not None and tokenizer is not None else "error"
    detail = None if status == "ok" else "model_not_loaded"
    return {"status": status, "detail": detail}

@app.post("/tts")
def tts(req: TTSRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")
    try:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform
        waveform = output.squeeze(0).cpu().numpy()
        waveform_int16 = (waveform * 32767.0).astype("int16")
        sample_rate = model.config.sampling_rate
        buffer = io.BytesIO()
        wavfile.write(buffer, rate=sample_rate, data=waveform_int16)
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="tts.wav"'},
        )
    except Exception as e:
        print("Error during TTS:", e)
        raise HTTPException(status_code=500, detail="TTS generation failed")
