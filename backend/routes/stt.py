# routes/stt.py
# from fastapi import APIRouter
# from services.tts_stt import recognize_from_microphone

# router = APIRouter()

# @router.get("/mic")
# def transcribe_microphone():
#     try:
#         text = recognize_from_microphone()
#         return {"status": "success", "text": text}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# stt.py
from fastapi import APIRouter
import whisper
import speech_recognition as sr
import tempfile
import os

router = APIRouter()

# Whisper 모델 로딩 (최초 1회)
model = whisper.load_model("small")
recognizer = sr.Recognizer()

@router.get("/mic")
def transcribe_microphone():
    try:
        with sr.Microphone(sample_rate=16000) as source:
            print("🎙 듣는 중... 말하세요.")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            print("🔇 발화 종료, 변환 중...")

        # 임시 WAV 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_data = audio.get_wav_data()
            f.write(wav_data)
            temp_audio_path = f.name

        result = model.transcribe(temp_audio_path, language="ko")
        os.remove(temp_audio_path)

        return {"status": "success", "text": result["text"]}

    except Exception as e:
        return {"status": "error", "message": str(e)}
