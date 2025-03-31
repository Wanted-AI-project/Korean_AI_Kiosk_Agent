from fastapi import APIRouter, HTTPException
from database import get_db_connection

router = APIRouter()

# ✅ 전체 메뉴 조회 API
@router.get("/")
def get_all_menus():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM menus")
    rows = cursor.fetchall()

    # Row → Dict 변환
    menu_list = [dict(row) for row in rows]

    conn.close()
    return {"menus": menu_list}

# ✅ 특정 메뉴 상세 조회 (menu_id 기준)
@router.get("/{menu_id}")
def get_menu_detail(menu_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM menus WHERE menu_id = ?", (menu_id,))
    row = cursor.fetchone()

    if row:
        return dict(row)
    raise HTTPException(status_code=404, detail="해당 메뉴를 찾을 수 없습니다.")

# ✅ 메뉴 이름으로 조회 API
@router.get("/name/{menu_name}")
def get_menu_by_name(menu_name: str):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM menus WHERE menu_name LIKE ?", (f"%{menu_name}%",))
    rows = cursor.fetchall()

    if rows:
        menu_list = [dict(row) for row in rows]
        return {"menus": menu_list}
    raise HTTPException(status_code=404, detail="해당 이름의 메뉴를 찾을 수 없습니다.")
