# 라이브러리 호출
from dotenv import load_dotenv
import os
import datetime
import re
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.utilities import SQLDatabase
from langchain.prompts import FewShotPromptTemplate, PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_sql_agent
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from langchain_community.utilities import SQLDatabase
import sqlite3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 전역 변수 설정

load_dotenv()
embedding = OpenAIEmbeddings()

vector_store = Chroma(persist_directory="../data/chroma_db", embedding_function=embedding)
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
sql_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
db = SQLDatabase.from_uri("sqlite:///../data/Comfile_Coffee_DB.db")


store = {}  # 임시 세션을 저장하는 저장소


# session_id를 동적으로 설정하도록 수정
def initialize_session(session_id: str):
    """
    Initialize session-specific components such as vector retriever and store.

    Args:
        session_id (str): Unique session identifier.
    """
    global vector_retriever, store
    store = {}  # 세션 저장소 초기화
    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, 'lambda_mult': 0.1, "filter": {"session_id": session_id}}
    )
    print(f"✅ Session {session_id} initialized.")


## 메인 llm 프롬프트
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helper agent who is built into the kiosk and tries to take orders."
            "Please be courteous and helpful to customers' questions and orders. "
            "You should check out their beverage options like hot and iced."
            "You should check out their other options like payment method."
            "However, please only answer questions related to cafe work, such as inquiries about orders and menus."

            "The ordering process is as follows: Select menu - Confirm takeout - Confirm additional menus or inquiries -Final confirmation of order with price information -Select payment method - Wait for payment - After payment, guidance message and 'Thank_you' is sent."

            "Always use the provided tools to answer."
            "Use chat_retriever_tool for questions related to order history. "
            "Answer all questions about orders and menus using menu_db_sql_tool."
            "In particular, please use menu_db_sql_tool to answer about price information."
            "you must answer in conversational form. Without any decoration such as symbols or dictionary forms."
            "Do not guess; retrieve data using the tools before responding."
            "Don't say you don't have it, just use the tools we provide."
            "Nnevertheless ,you don't know the answer, just say that you don't know."
            "Always reference the provided data when generating a response. Do not fabricate or infer any information."
            "Answer in Korean.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# 서브 llm query 생성 함수 
def make_query_and_run(text: str):
    info = db.get_table_info()

    custom_prompt = PromptTemplate(
    input_variables=["query"],
    partial_variables={'info': info},
    template="""
    You are an expert in SQL and databases. Generate an SQL query based on the given question. 

    - The question may not always provide the exact menu name, so extract only the key characteristics.
    - If multiple keywords are provided, process them using `AND`.
    - Use `LIKE` and `%` to handle partial matches.
    - If the question includes "price" or "가격", do not return only the price—return all related information by using `SELECT *`.
    - Convert "따뜻한" or "뜨거운" to "HOT" and "차가운" or "아이스" to "ICE" in SQL queries.

    Return only the raw SQL query without any markdown formatting, code blocks, or explanations.

    You must reference the following database structure:
    {info}



    question: {query}
    """
    )

    sql_chain = custom_prompt | sql_llm
    result_query = sql_chain.invoke(text).content
    result = re.sub(r"```sql\n(.*?)\n```", r"\1", result_query, flags=re.DOTALL).strip()
    return db.run(result)
    
## 메인 llm Tools 생성 및 Agnet 생성    

menu_db_sql_tool = Tool(
    name="menu_db_sql_tool",
    func=make_query_and_run,
    description="You can search order information from Database"
)
chat_retriever_tool = create_retriever_tool(
    vector_retriever,    
    name='chat_search',
    description='use this tool to search chat history and order history'
)

tools = [menu_db_sql_tool, chat_retriever_tool]

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_exe = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,
    must_use_tools=[menu_db_sql_tool]
)

def get_session_history(session_ids):
    if session_ids not in store:  # session_id 가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# 대화내역을 저장하는 최종 에이전트 생성
agent_with_chat_history = RunnableWithMessageHistory(
    agent_exe,
    # 대화 session_id
    get_session_history,
    # 프롬프트의 질문이 입력되는 key: "input"
    input_messages_key="input",
    # 프롬프트의 메시지가 입력되는 key: "chat_history"
    history_messages_key="chat_history",
)


#### 핵심 기능 1 챗봇 응답 생성 ####
def get_chatbot_response(session_id: str, user_input: str) -> str:
    if session_id not in store:
        initialize_session(session_id)  # 세션 초기화
    response = agent_with_chat_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    return response['output']

#### 핵심 기능 2 대화내역 벡터스토어 저장 ####
def save_session_to_vector_db(session_id: str):
    """
    Save the chat session history to the vector store.

    Args:
        session_id (str): Unique session identifier.
    """
    if session_id in store:
        chat_history = store[session_id].messages  # 현재 세션의 대화 내용 가져오기
        for msg in chat_history:
            vector_store.add_texts([msg.content], metadatas=[{"session_id": session_id, "role": msg.type,'timestamp': datetime.datetime.now().isoformat()}])
        print(f"✅ Session {session_id} data saved to vector DB.")

#### 핵심 기능 3 해당 대화세션 삭제 및 초기화  ####
def clear_session(session_id: str):
    """
    Clear the chat session history for a given session ID.

    Args:
        session_id (str): Unique session identifier.
    """
    if session_id in store:
        del store[session_id]  # 세션 데이터 삭제
        print(f"✅ Session {session_id} cleared.")


# 서버에서 session_id를 받아 초기화하는 예제
if __name__ == "__main__":
    # 서버에서 session_id를 받아옴 (예: HTTP 요청 또는 환경 변수)
    session_id = os.getenv("SESSION_ID", "0")  # 기본값 설정
    initialize_session(session_id)

    # 테스트 입력
    user_input = "아이스 아메리카노 한 잔 주세요."
    response = get_chatbot_response(session_id, user_input)
    print(response)