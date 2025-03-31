from fastapi import APIRouter, HTTPException, Depends
from database import get_db_connection
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# Pydantic 모델 정의
class User(BaseModel):
    name: str
    phone: str
    face_encoding: Optional[str] = None  # 얼굴 인식 데이터 (선택적)
    

# ✅ 1️⃣ **전체 사용자 목록 조회 (GET)**
@router.get("/")
def get_users():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    conn.close()

    return {"users": [dict(user) for user in users]}


# ✅ 2️⃣ **특정 사용자 조회 (GET) - user_id 기반**
@router.get("/{user_id}")
def get_user(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if user is None:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    return dict(user)


# ✅ 3️⃣ **새로운 사용자 등록 (POST)**
@router.post("/")
def create_user(user: User):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO users (name, phone, face_encoding, created_at) VALUES (?, ?, ?, datetime('now'))",
        (user.name, user.phone, user.face_encoding),
    )
    conn.commit()
    conn.close()

    return {"message": "사용자가 성공적으로 등록되었습니다."}


# ✅ 4️⃣ **사용자 정보 업데이트 (PUT)**
@router.put("/{user_id}")
def update_user(user_id: int, user: User):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE users SET name = ?, phone = ?, face_encoding = ? WHERE user_id = ?",
        (user.name, user.phone, user.face_encoding, user_id),
    )
    conn.commit()
    conn.close()

    return {"message": f"사용자 {user_id} 정보가 업데이트되었습니다."}


# ✅ 5️⃣ **사용자 삭제 (DELETE)**
@router.delete("/{user_id}")
def delete_user(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

    return {"message": f"사용자 {user_id}가 삭제되었습니다."}

# ✅ ** face_detection 로그인 ** 
@router.post("/login")
def login_user_face(data: dict):
    print("얼굴 인식된 사용자:", data["name"])
    return {"message": f"{data['name']} 로그인 처리됨"}

