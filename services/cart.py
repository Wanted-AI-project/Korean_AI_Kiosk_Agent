### 데이터베이스 관련 라이브러리 호출###
import sqlite3
from sqlite3 import Error
import os
from dotenv import load_dotenv
import re
import datetime
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


# llm = ChatGoogleGenerativeAI(model_name="gemini-2.0.flash",temperature=0)
# conn = sqlite3.connect("/../data/Comfile_Coffee_DB.db")
# cursor = conn.cursor()

### 1. 장바구니 서비스 구현하기 ###
class Cart:
    def __init__(self):
        """
        Initialize an
        """
        self.items = []  # 장바구니 항목 리스트

    def add_item(self, item_name: str, price: float, image:str, quantity: int =1):
        """
        Add an item to the cart.

        Args:
            item_name (str): The name of the item.
            price (float): The price of the item.
            quantity (int): The quantity of the item. Default is 1.
        """
        for item in self.items:
            if item['item'] == item_name:
                item['count'] += quantity
                return
        self.items.append({"item": item_name, "price": price, "image": image ,"count": quantity})

    def remove_item(self, item_name: str):
        """
        Remove an item from the cart.

        Args:
            item_name (str): The name of the item to remove.
        """
        self.items = [item for item in self.items if item['name'] != item_name]

    def update_quantity(self, item_name: str, quantity: int):
        """
        Update the quantity of an item in the cart.

        Args:
            item_name (str): The name of the item.
            quantity (int): The new quantity. If 0, the item will be removed.
        """
        for item in self.items:
            if item['item'] == item_name:
                if quantity == 0:
                    self.remove_item(item_name)
                else:
                    item['count'] = quantity
                return

    def calculate_total(self) -> float:
        """
        Calculate the total price of all items in the cart.

        Returns:
            float: The total price.
        """
        return sum(item['price'] * item['count'] for item in self.items)

    def clear_cart(self):
        """
        Clear all items from the cart.
        """
        self.items = []

    def get_cart_items(self):
        """
        Get a list of all items in the cart.

        Returns:
            list: A list of items in the cart.
        """
        return self.items
    
    def save_to_database(self):
        """
        현재 장바구니 정보를 DB의 주문내역으로 저장한다.
        """ 


        return "DB에 저장되었습니다."


### 자연어 명령 파싱 ###
def parse_with_gemini(command: str):
    """
    Parse a natural language command to perform actions on the cart.

    Args:
        command (str): STT로 받은 자연어 명령.

    Returns:
        json: 자연어 음성으로부터 응답 결과.
    """
    text=command
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="다음 문장에서 목적 ,메뉴이름, 수량 JSON 형식으로 추출해 {text}"
    )


