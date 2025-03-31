from fastapi import FastAPI
from routes import users, chatbot, menu, stt
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI 키오스크 백엔드",
    description="ComposeCoffee의 AI 키오스크의 백엔드"
    )

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중에는 일단 전체 허용, 나중에 프론트 서버 확정되면 수정
    allow_methods=["*"],
    allow_headers=["*"],
)

# 기능별 라우터 등록
app.include_router(users.router, prefix="/users")
app.include_router(chatbot.router, prefix="/chatbot")
app.include_router(menu.router, prefix="/menu")
app.include_router(stt.router, prefix="/stt")

@app.get("/")
def root():
    return {"message": "AI 키오스크 FastAPI 서버 작동 중!"}
