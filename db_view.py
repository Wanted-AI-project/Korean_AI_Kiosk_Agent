import sqlite3
import pandas as pd
import pickle
import numpy as np

# 데이터베이스 경로 설정
DB_PATH = "../data/Comfile_Coffee_DB.db"

def view_database():
    """데이터베이스 내용 조회"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 테이블 목록 확인
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"📌 존재하는 테이블: {tables}")

    # users 테이블 데이터 조회
    cursor.execute("SELECT user_id, name FROM users")
    users = cursor.fetchall()
    
    print("\n📌 등록된 사용자 목록:")
    for user_id, name in users:
        print(f"ID: {user_id}, 이름: {name}")
    
    conn.close()

def delete_user(user_id):
    """특정 사용자 삭제"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
        print(f"✅ ID {user_id} 사용자 삭제 완료")
    except Exception as e:
        print(f"❌ 삭제 중 오류 발생: {e}")
    finally:
        conn.close()

def clear_database():
    """전체 데이터베이스 초기화"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM users")
        conn.commit()
        print("✅ 데이터베이스 초기화 완료")
    except Exception as e:
        print(f"❌ 초기화 중 오류 발생: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    while True:
        print("\n=== 데이터베이스 관리 메뉴 ===")
        print("1. 데이터베이스 내용 조회")
        print("2. 특정 사용자 삭제")
        print("3. 전체 데이터베이스 초기화")
        print("4. 종료")
        
        choice = input("\n선택하세요 (1-4): ")
        
        if choice == "1":
            view_database()
        elif choice == "2":
            user_id = input("삭제할 사용자 ID를 입력하세요: ")
            try:
                delete_user(int(user_id))
            except ValueError:
                print("❌ 올바른 ID를 입력하세요")
        elif choice == "3":
            confirm = input("정말로 모든 데이터를 삭제하시겠습니까? (y/n): ")
            if confirm.lower() == 'y':
                clear_database()
            else:
                print("❌ 취소되었습니다")
        elif choice == "4":
            print("프로그램을 종료합니다")
            break
        else:
            print("❌ 잘못된 선택입니다")

# 확인 = python db_test.py