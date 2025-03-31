# routes/chatbot.py

from fastapi import APIRouter, Query
from pydantic import BaseModel

# chatbot_origin.py의 핵심 로직을 import
from module.chatbot_origin import (
    get_chatbot_response,
    save_session_to_vector_db,
    clear_session,
    initialize_session
)

router = APIRouter()


# ✅ 1. 사용자 챗봇 질의 처리 (POST)
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

@router.post("/chat")
def chatbot_reply(data: ChatRequest):
    """
    사용자의 입력에 대해 챗봇 응답을 생성합니다.
    """
    response = get_chatbot_response(session_id=data.session_id, user_input=data.user_input)
    return {"response": response}


# ✅ 2. 세션 대화 이력 벡터 DB 저장 (POST)
@router.post("/save-session")
def save_session(session_id: str = Query(..., description="저장할 세션 ID")):
    """
    현재 세션의 대화 내용을 벡터 DB에 저장합니다.
    """
    save_session_to_vector_db(session_id)
    return {"status": "success", "message": f"{session_id} 대화 내용 저장 완료"}


# ✅ 3. 세션 초기화 및 삭제 (DELETE)
@router.delete("/clear-session")
def clear_chat_session(session_id: str = Query(..., description="초기화할 세션 ID")):
    """
    세션 내 대화 이력 초기화 및 삭제합니다.
    """
    clear_session(session_id)
    return {"status": "success", "message": f"{session_id} 세션 초기화 완료"}


# ✅ 4. 명시적 세션 초기화 (GET)
@router.get("/initialize-session")
def initialize_chatbot_session(session_id: str = Query(..., description="초기화할 세션 ID")):
    """
    세션을 명시적으로 초기화합니다.
    """
    initialize_session(session_id)
    return {"status": "success", "message": f"{session_id} 세션 초기화 완료"}
